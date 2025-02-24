#!/usr/bin/env python3
# app.py

import os
import csv
import cgi
from datetime import datetime
from wsgiref.simple_server import make_server

# 70 survey questions for incoming college students
# focusing on their high school experiences and admissions process.
QUESTIONS = [
    "How satisfied were you with the overall academic experience in high school?",
    "Did you feel your high school provided adequate college counseling?",
    "How well did your high school teachers prepare you for college-level coursework?",
    "Did you feel supported by your high school guidance counselor during the admissions process?",
    "How stressful was the college application process for you?",
    "How confident are you that your high school experience prepared you for college?",
    "Did you have sufficient access to advanced or AP classes in high school?",
    "How often did you miss classes in high school?",
    "Did you feel your high school workload was manageable?",
    "How supportive was your family during your college application process?",
    "Did you receive help from mentors or advisors when choosing colleges to apply to?",
    "How comfortable are you with the idea of transitioning from high school to college?",
    "Did you participate in extracurricular activities in high school that influenced your college choice?",
    "How confident are you in your ability to handle college-level reading assignments?",
    "Did you feel peer pressure in high school regarding college choices?",
    "How satisfied are you with your chosen college’s location?",
    "How stressful was completing financial aid forms (e.g., FAFSA)?",
    "Did you feel your high school adequately explained scholarship opportunities?",
    "Were you satisfied with your standardized test (SAT/ACT) preparation in high school?",
    "How supported did you feel when writing college application essays?",
    "How well did your high school teach you time-management skills?",
    "Did you have a clear career path in mind before applying to college?",
    "How effective were your high school’s resources (e.g., library, tutoring) for academic success?",
    "Did your high school environment encourage you to pursue higher education?",
    "How comfortable are you with using online tools for college applications and course registration?",
    "Do you worry about making friends in college?",
    "Did you feel your high school environment was too competitive, not competitive enough, or just right?",
    "How concerned are you about the cost of attending college?",
    "Were you satisfied with your high school’s approach to diversity and inclusion?",
    "Did you receive enough information about dorm or housing options at your college?",
    "How do you feel about living away from home (if applicable) for college?",
    "Did your high school teachers encourage critical thinking and problem-solving?",
    "How confident are you in your ability to manage finances in college?",
    "Did you find the college admissions process confusing?",
    "How often did you meet with your high school counselor about college plans?",
    "Did you have access to adequate mental health resources in high school?",
    "Are you concerned about balancing social life and academics in college?",
    "How well do you think your high school’s grading standards prepared you for college?",
    "Did you attend any college fairs or admissions events through your high school?",
    "How worried are you about meeting new academic standards in college?",
    "Did you feel your high school offered enough extracurricular leadership opportunities?",
    "How well did your high school support you in exploring different majors or career paths?",
    "Did you feel pressured to apply to certain colleges by peers or family?",
    "How prepared are you to handle more freedom and responsibility in college?",
    "Did you receive help in crafting your personal statement or admissions essays?",
    "How concerned are you about the pace of college semesters or quarters compared to high school?",
    "Did your high school discuss potential culture shock or diversity at college?",
    "How well did your high school prepare you for independent study or research?",
    "Are you worried about handling college-level math or science requirements?",
    "Did your high school encourage you to visit college campuses before deciding?",
    "How confident are you that you chose the right college for your goals?",
    "Did you have to balance part-time work with your high school studies?",
    "How effective was your high school at teaching study skills for college success?",
    "Did you feel your high school offered enough AP/IB/Honors courses (if available)?",
    "How anxious are you about meeting new people and making friends in college?",
    "Do you believe your high school discipline or attendance policies affected your college readiness?",
    "Did you research multiple colleges thoroughly before deciding where to enroll?",
    "How well did your high school accommodate your learning style or needs?",
    "Are you concerned about homesickness when starting college?",
    "Did your high school provide adequate technology training for college tasks (e.g., writing papers, presentations)?",
    "How would you rate your high school’s communication about admissions deadlines and requirements?",
    "Are you worried about balancing extracurriculars with academics in college?",
    "Did your high school environment foster independence and self-motivation?",
    "How do you feel about the orientation program offered by your college?",
    "Did you feel your high school environment was safe and conducive to learning?",
    "Are you concerned about the size of your chosen college (too big or too small)?",
    "How prepared do you feel to handle any financial aid or scholarship renewal processes?",
    "Did you have enough guidance choosing a major or deciding to be undecided?",
    "Are you satisfied with the admissions decision you received from this college?",
    "Do you have concerns about transferring if this college does not meet your expectations?",
    "How confident are you in applying high school study habits to college-level courses?",
    "Do you believe your high school experiences will positively influence your college success?",
    "How excited are you to start this new chapter of your academic journey?"
]

def build_survey_page():
    """
    Returns an HTML page with:
      - Title: "Student survey"
      - Instructions: "1 means doesn't agree, 5 means extremely agree"
      - Fields for student_name, student_id, student_email
      - 70 questions with radio buttons
    """
    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'>")
    html_parts.append("<head>")
    html_parts.append("  <meta charset='UTF-8'>")
    html_parts.append("  <title>Student survey</title>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("  <h1>Student survey</h1>")
    html_parts.append("  <p>1 means doesn't agree, 5 means extremely agree.</p>")
    
    # Student identification fields
    html_parts.append("  <form method='POST' action='/submit'>")
    html_parts.append("    <label>Student Name:</label><br>")
    html_parts.append("    <input type='text' name='student_name' required><br><br>")
    
    html_parts.append("    <label>Student ID:</label><br>")
    html_parts.append("    <input type='text' name='student_id' required><br><br>")
    
    html_parts.append("    <label>Student Email:</label><br>")
    html_parts.append("    <input type='email' name='student_email' required><br><br>")

    # Build each question with radio buttons
    for i, question in enumerate(QUESTIONS, start=1):
        html_parts.append(f"    <div>")
        html_parts.append(f"      <label><strong>Question {i}:</strong> {question}</label><br>")
        for val in range(1, 6):
            html_parts.append(f"      <input type='radio' name='q{i}' value='{val}' required>")
            html_parts.append(f"      <label>{val}</label>")
        html_parts.append(f"    </div>")
        html_parts.append(f"    <br>")

    html_parts.append("    <input type='submit' value='Submit Survey'>")
    html_parts.append("  </form>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)

def build_thank_you_page():
    """Returns a simple HTML thank-you message."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Survey Submission</title>
</head>
<body>
  <h2>Thank you for submitting the survey!</h2>
</body>
</html>
"""

def handle_survey(environ):
    """
    Handles the GET request for the survey page.
    """
    response_body = build_survey_page()
    return ("200 OK", [("Content-Type", "text/html")], response_body)

def handle_submit(environ):
    """
    Handles the POST request to submit the survey.
    Parses form data, writes to CSV, and returns a thank-you page.
    """
    try:
        # Parse form data
        form = cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=True)

        # Build responses dict
        responses = {}

        # Student identification fields
        student_name = form.getvalue("student_name", "")
        student_id = form.getvalue("student_id", "")
        student_email = form.getvalue("student_email", "")
        responses["student_name"] = student_name
        responses["student_id"] = student_id
        responses["student_email"] = student_email

        # Survey responses
        for i in range(1, len(QUESTIONS) + 1):
            key = f"q{i}"
            value = form.getvalue(key, None)
            responses[key] = value

        # Add a timestamp
        responses["timestamp"] = datetime.now().isoformat()

        # Define field order for CSV
        fieldnames = [
            "student_name",
            "student_id",
            "student_email"
        ] + [f"q{i}" for i in range(1, len(QUESTIONS) + 1)] + ["timestamp"]

        csv_file = "student_data.csv"
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(responses)

        # Return thank-you page
        response_body = build_thank_you_page()
        return ("200 OK", [("Content-Type", "text/html")], response_body)

    except Exception as e:
        # In case of any error, return a 500 page
        return ("500 Internal Server Error", [("Content-Type", "text/plain")], str(e))

def application(environ, start_response):
    """
    A simple WSGI application to serve the survey (GET /)
    and handle submissions (POST /submit).
    """
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    if path == "/" and method == "GET":
        status, headers, body = handle_survey(environ)
    elif path == "/submit" and method == "POST":
        status, headers, body = handle_submit(environ)
    else:
        # Return 404 for any other route
        status = "404 Not Found"
        headers = [("Content-Type", "text/plain")]
        body = "Page Not Found"

    start_response(status, headers)
    # If body is a string, convert to bytes
    if isinstance(body, str):
        body = body.encode("utf-8")
    return [body]

def run_server():
    """
    Runs the WSGI application using Python's built-in wsgiref.simple_server
    but binds to a specific IP address instead of 0.0.0.0 or localhost.
    
    For production, consider a more robust WSGI server (gunicorn, uWSGI, etc.).
    """
    # Replace '192.168.0.157' with your desired IP address on your network.
    host = "192.168.0.157"
    port = 5000
    
    httpd = make_server(host, port, application)
    print(f"Serving on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()

