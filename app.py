import os
import cv2
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    send_file,
    session,
    redirect,
    url_for,
    flash
)
from omr_detection import process_omr

from models import db, OMRSheet, OMRAnswer, AnswerKey
from helpers import build_excel

app = Flask(__name__)
app.secret_key = "omr-secret-key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///omr.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    answer_key = {
        ak.question_no: ak.correct_option
        for ak in AnswerKey.query.all()
    }

    total_questions = max(150, len(answer_key))

    if request.method == "POST":
        files = request.files.getlist("files")
        results = {}
        latest_sheet_ids = []

        for file in files:
            if file.filename == "":
                continue

            try:
                original_name = file.filename
                temp_path = os.path.join(UPLOAD_FOLDER, "temp_" + original_name)
                file.save(temp_path)

                img = cv2.imread(temp_path)
                if img is None:
                    raise Exception("Invalid image")

                data = process_omr(img)

                name = data["name"]
                class_name = data["class_name"]
                section = data["section"]
                roll_number = data["roll_number"]
                stream = data["stream"]
                set_number = data["set_number"]
                subject_code = data["subject_code"]
                admission_number = data["admission_number"]
                answers = data["answers"]
                
                final_answers = []

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

                    final_answers.append({
                        "value": selected,
                        "is_correct": is_correct
                    })
                
                name_only, ext = os.path.splitext(original_name)

                new_filename = f"{roll_number}_{name_only}{ext}"
                new_path = os.path.join(UPLOAD_FOLDER, new_filename)

                if os.path.exists(new_path):
                    os.remove(new_path)

                os.rename(temp_path, new_path)


                sheet = OMRSheet.query.filter_by(
                    roll_number=roll_number
                ).first()

                if not sheet:
                    sheet = OMRSheet(roll_number=roll_number)
                    db.session.add(sheet)
                    db.session.flush()
                else:
                    OMRAnswer.query.filter_by(sheet_id=sheet.id).delete()

                final_answer = []

                for i, ans in enumerate(answers):
                    q_no = f"Q{str(i+1).zfill(3)}"

                    if ans == "-":
                        selected = "Empty"
                    elif ans == "MULTI":
                        selected = "Multiple"
                    else:
                        selected = ans

                    correct = answer_key.get(q_no)
                    is_correct = selected == correct

                    db.session.add(
                        OMRAnswer(
                            question_no=q_no,
                            selected_option=selected,
                            is_correct=is_correct,
                            sheet_id=sheet.id
                        )
                    )

                    final_answer.append({
                        "value": selected,
                        "is_correct": is_correct
                    })

                correct_count = sum(1 for a in final_answer if a["is_correct"])
                total = len(final_answer)
                wrong = total - correct_count
                percentage = (correct_count / total * 100) if total else 0

                sheet.original_file_name = original_name
                sheet.result_file = new_filename

                sheet.name = name
                sheet.class_name = class_name
                sheet.section = section
                sheet.stream = stream

                sheet.set_number = set_number
                sheet.subject_code = subject_code
                sheet.admission_number = admission_number

                sheet.total_questions = total
                sheet.correct_answers = correct_count
                sheet.wrong_answers = wrong
                sheet.percentage = percentage

                db.session.commit()

                results[new_filename] = {
                    "name": name,
                    "roll_number": roll_number,
                    "class": class_name,
                    "section": section,
                    "stream": stream,
                    "answers": final_answers,
                    "percentage": round((sum(a["is_correct"] for a in final_answers) / len(final_answers) * 100) if final_answers else 0, 2)
                }

                latest_sheet_ids.append(sheet.id)

            except Exception as e:
                results[file.filename] = f"Error: {str(e)}"

        session["latest_sheet_ids"] = latest_sheet_ids
        
    answer_key = {
        ak.question_no: ak.correct_option
        for ak in AnswerKey.query.all()
    }


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
    ids = session.get("latest_sheet_ids", [])

    if not ids:
        return {"message": "No recent data"}, 404

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