"""Use this script to sync the local notebooks to their Google colab locations."""


import json
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io


SCOPES = ['https://www.googleapis.com/auth/drive']


def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds


def update_notebook(drive_service, file_id, local_notebook_path):
    with open(local_notebook_path, 'r', encoding='utf-8') as f:
        notebook_json = f.read()

    media_body = MediaIoBaseUpload(io.BytesIO(notebook_json.encode('utf-8')), mimetype='application/ipynb', resumable=True)

    updated_file = drive_service.files().update(
        fileId=file_id,
        media_body=media_body
    ).execute()

    print(f"Updated {updated_file['name']} ({file_id})")


def main():
    creds = authenticate()
    drive_service = build('drive', 'v3', credentials=creds)

    file_mapping = {
        'lecture_notes/Week_1.ipynb': '1P4PeoNnpbKffAMEELvzxllDPYoBLIRul',
        'lecture_notes/Week_2.ipynb': '1zySxeu4ykgltnk8oVkeEazoqJ6ET4pZo',
        'lecture_notes/Week_3.ipynb': '1jB1_uX943FWZ47_rMoIyMUMHZ1MjUo4Z',
        'lecture_notes/Week_4.ipynb': '1GJrEeh0Y4fY_xElSaSJqiOKty1VM8Hu3',
        'lecture_notes/Week_5.ipynb': '1j320zGPJMl1imDznSUceVFB7n41lzWU3', 
        'lecture_notes/Week_6.ipynb': '1u2FkH2Vss8K93XDEyIu-9YwX8JYa8jkS', 
        'lecture_notes/Week_7.ipynb': '1-b0Xy1b8jd0x8T8LQmfoY1Ha1Y3L9HkC', 
        'lecture_notes/Week_8.ipynb': '1dll9QYaQKSa0A7p1eYcdHGjxKpySu0UD', 
        'lecture_notes/Week_9.ipynb': '1oXQ05dB3z0qsxAvKbeIky3sL5C1vKhcO', 
        'lecture_notes/Week10.ipynb': '11n1RRV1qnSgzfuagqFFgZ5V7ZV-mVT6h',
    }

    for local_path, file_id in file_mapping.items():
        update_notebook(drive_service, file_id, local_path)


if __name__ == '__main__':
    main()

