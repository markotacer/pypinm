import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd

# Konfiguracija Google API-ja
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Datoteka z dovoljenji

@st.cache_data
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)

# Prikaz prijave in dostopa do Google Drive
st.title("Wine Orders App")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if st.button("Login with Google"):
    st.session_state.authenticated = True

if st.session_state.authenticated:
    st.success("Prijavljen!")
    
    # Pridobi datoteke iz mape "VINO"
    drive_service = get_drive_service()
    results = drive_service.files().list(q="name contains 'VINO'").execute()
    files = results.get('files', [])
    
    if not files:
        st.error("Mapa 'VINO' ni najdena!")
    else:
        st.write("Najdene datoteke:")
        for file in files:
            st.write(f"{file['name']} - {file['id']}")
        
        # Branje naročil iz prve tabele (test)
        file_id = files[0]['id']  # Zaenkrat vzamemo prvo datoteko
        df = pd.read_csv(f'https://drive.google.com/uc?id={file_id}')  # Pretvori v DataFrame
        st.dataframe(df)
