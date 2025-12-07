import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
from ...config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_FROM_NAME

DEMO_ACCOUNTS = ["test@example.com", "test2@example.com"]
DEMO_ACCOUNT_OTP = "123456"

def is_demo_account(email: str) -> bool:
    """Check if email is a demo/test account"""
    return email.lower() in [acc.lower() for acc in DEMO_ACCOUNTS]

def send_email(to_email: str, subject: str, body: str):
    msg = MIMEMultipart()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, to_email, msg.as_string())

def generate_otp():
    return str(random.randint(100000, 999999)) 
