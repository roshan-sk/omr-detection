from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class OMRSheet(db.Model):
    __tablename__ = 'omr_sheet'

    id = db.Column(db.Integer, primary_key=True)

    original_file_name = db.Column(db.String(200))
    result_file = db.Column(db.String(200), unique=True)
    
    name = db.Column(db.String(40), index=True)
    roll_number = db.Column(db.String(50), index=True, unique=True)
    class_name = db.Column(db.String(10))
    section = db.Column(db.String(10))
    stream = db.Column(db.String(20))

    set_number = db.Column(db.String(10))
    subject_code = db.Column(db.String(20))
    admission_number = db.Column(db.String(10))

    total_questions = db.Column(db.Integer, default=0)
    correct_answers = db.Column(db.Integer, default=0)
    wrong_answers = db.Column(db.Integer, default=0)
    percentage = db.Column(db.Float, default=0)

    uploaded_time = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )

    answers = db.relationship(
        'OMRAnswer',
        backref='sheet',
        cascade="all, delete-orphan",
        lazy=True
    )



class OMRAnswer(db.Model):
    __tablename__ = 'omr_answer'

    id = db.Column(db.Integer, primary_key=True)

    question_no = db.Column(db.String(10), nullable=False)
    selected_option = db.Column(db.String(10))
    is_correct = db.Column(db.Boolean)
    
    created_at = db.Column(
        db.DateTime,
        server_default=db.func.current_timestamp()
    )

    sheet_id = db.Column(
        db.Integer,
        db.ForeignKey('omr_sheet.id', ondelete="CASCADE"),
        nullable=False
    )



class AnswerKey(db.Model):
    __tablename__ = 'answer_key'

    id = db.Column(db.Integer, primary_key=True)

    question_no = db.Column(db.String(10), unique=True, nullable=False)
    correct_option = db.Column(db.String(5), nullable=False)
