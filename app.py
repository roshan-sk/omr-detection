import os
import cv2
import zipfile
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect
)

from omr_detection import process_omr
from models import db, OMRSheet, AnswerKey
from helpers import build_excel
import uuid
import time
from flask import Flask
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv


load_dotenv()

app = Flask(__name__)
app.secret_key = "omr-secret-key"


db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise ValueError("DATABASE_URL is not set")

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def process_single_file_worker(path, original_name, answer_key, batch_id):
    try:
        img = cv2.imread(path)
        if img is None:
            raise Exception("Invalid image")

        data = process_omr(img)

        roll_number = data["roll_number"]

        unique_id = str(uuid.uuid4())[:8]
        name_only, ext = os.path.splitext(original_name)
        new_filename = f"{roll_number}_{name_only}_{unique_id}{ext}"
        new_path = os.path.join("uploads", new_filename)

        # os.rename(path, new_path)

        answers_json = {}
        final_answer = []

        for i, ans in enumerate(data["answers"]):
            q_no = f"Q{str(i+1).zfill(3)}"

            if ans == "-":
                selected = "Empty"
            elif ans == "MULTI":
                selected = "Multiple"
            else:
                selected = ans

            correct = answer_key.get(q_no)
            is_correct = selected == correct

            answers_json[q_no] = {
                "selected": selected,
                "is_correct": is_correct
            }

            final_answer.append({
                "value": selected,
                "is_correct": is_correct
            })

        correct_count = sum(1 for a in final_answer if a["is_correct"])
        total = len(final_answer)
        wrong = total - correct_count
        percentage = (correct_count / total * 100) if total else 0

        return {
            "sheet_data": {
                "roll_number": data["roll_number"],
                "original_file_name": original_name,
                "result_file": new_filename,
                "name": data["name"],
                "class_name": data["class_name"],
                "section": data["section"],
                "stream": data["stream"],
                "set_number": data["set_number"],
                "subject_code": data["subject_code"],
                "admission_number": data["admission_number"],
                "total_questions": total,
                "correct_answers": correct_count,
                "wrong_answers": wrong,
                "percentage": percentage,
                "answers": answers_json,
                "batch_id": batch_id
            },
            "result": {
                "name": data["name"],
                "roll_number": data["roll_number"],
                "class": data["class_name"],
                "section": data["section"],
                "stream": data["stream"],
                "answers": final_answer,
                "percentage": round(percentage, 2)
            },
            "key": new_filename
        }

    except Exception as e:
        return {
            "error": f"{original_name}: {str(e)}"
        }


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    answer_key = {
        ak.question_no: ak.correct_option
        for ak in AnswerKey.query.all()
    }

    total_questions = max(150, len(answer_key))

    if request.method == "POST":
        start_time = time.time()

        files = request.files.getlist("files")
        results = {}
        all_sheets = []
        tasks = []

        file_paths = []
        batch_id = str(uuid.uuid4())


        for file in files:
            if not file.filename:
                continue

            if file.filename.endswith(".zip"):
                zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(zip_path)

                extract_folder = os.path.join(UPLOAD_FOLDER, file.filename + "_extracted")
                os.makedirs(extract_folder, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    start = time.time()
                    zip_ref.extractall(extract_folder)
                    print("Zip extract time:", time.time() - start)

                for root, _, extracted_files in os.walk(extract_folder):
                    for fname in extracted_files:
                        file_paths.append((os.path.join(root, fname), fname))

            else:
                temp_path = os.path.join(UPLOAD_FOLDER, "temp_" + file.filename)
                file.save(temp_path)
                file_paths.append((temp_path, file.filename))

        with ProcessPoolExecutor(max_workers = max(2, os.cpu_count() - 1)) as executor:
            futures = [
                executor.submit(process_single_file_worker, path, name, answer_key, batch_id)
                for path, name in file_paths
            ]

            for future in as_completed(futures):
                result = future.result()

                if "error" in result:
                    results[result["error"]] = result["error"]
                    continue

                sheet = OMRSheet(**result["sheet_data"])
                all_sheets.append(sheet)

                results[result["key"]] = result["result"]

        # ✅ Single DB commit
        db.session.add_all(all_sheets)
        db.session.commit()

        session["latest_batch_id"] = batch_id

        print("Execution time:", time.time() - start_time)

    return render_template(
        "index.html",
        results=results,
        total_questions=total_questions,
        existing_keys=answer_key
    )


@app.route("/save_answer_key", methods=["POST"])
def save_answer_key():
    total = int(request.form.get("total_questions", 100))

    for i in range(1, total + 1):
        ans = request.form.get(f"q{i}")

        if ans:
            q_no = f"Q{str(i).zfill(3)}"

            existing = AnswerKey.query.filter_by(question_no=q_no).first()

            if existing:
                existing.correct_option = ans.upper()
            else:
                db.session.add(
                    AnswerKey(
                        question_no=q_no,
                        correct_option=ans.upper()
                    )
                )

    db.session.commit()
    return redirect("/")


@app.route("/api/export_latest")
def export_latest():
    batch_id = session.get("latest_batch_id")

    if not batch_id:
        return {"message": "No recent data"}, 404

    sheets = OMRSheet.query.filter_by(batch_id=batch_id).all()

    if not sheets:
        return {"message": "No records"}, 404

    ids = [s.id for s in sheets]

    data = build_excel(sheet_ids=ids)

    if not data:
        return {"message": "No records"}, 404

    output, filename = data

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)