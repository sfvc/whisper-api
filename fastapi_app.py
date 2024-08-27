from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import whisper
import torch
from typing import List
from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Verificar si hay una GPU NVIDIA disponible
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el modelo Whisper
model = whisper.load_model("base", device=DEVICE)

app = FastAPI()

# Configurar la clave de API de OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/whisper/")
async def handler(files: List[UploadFile] = File(...), form_format: str = Form(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se proporcionaron archivos")

    if not form_format:
        raise HTTPException(status_code=400, detail="No se proporcionó el formato del formulario")

    try:
        form_format_dict = json.loads(form_format)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON inválido para form_format")

    results = []

    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())
            
            result = model.transcribe(temp.name)
            transcribed_text = result['text']

            formatted_response = await process_and_format_text(transcribed_text, form_format_dict)

            results.append({
                'filename': file.filename,
                'transcript': formatted_response,
            })

    return JSONResponse(content={'results': results})

async def process_and_format_text(text: str, form_format: dict) -> dict:
    format_instructions = "\n".join([f"{key}: {description}" for key, description in form_format.items()])
    
    prompt = f"""
    Dado el siguiente texto transcrito: 

    {text}

    1. Corrige cualquier error gramatical o de ortografía en el texto.
    2. Extrae la información relevante y formateala según la siguiente estructura:

    {format_instructions}

    3. Devuelve el resultado como un diccionario JSON, asegurándote de que los campos coincidan exactamente con los nombres proporcionados en la estructura.
    4. Si alguna información no está presente en el texto, deja el campo correspondiente vacío.
    5. Asegúrate de que los campos en formato numerico se devuelvan como numeros.
    6. Asegurate que las palabras que hacen referencia a simpobolos o caracterez especiales se traduzcan correctamente al caracter correspondiente.
        Ejemplo: arroba es @
                 porcentaje es %
                 pesos es $
                 guion medio es -
                 guin bajo es _
        Y asi sucesivamente con todos los carateres y simbolos.
    7. Formatea las fechas de cualquier forma a YYYY/mm/dd
        Ejemplo: 21 de Agosto de 2024 deberia ser 2024-08-21

    Responde ÚNICAMENTE con el diccionario JSON, sin texto adicional.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en procesar y formatear texto. Tu tarea es corregir, extraer información y formatearla según las instrucciones proporcionadas."},
            {"role": "user", "content": prompt}
        ]
    )

    response_text = response.choices[0].message.content.strip()

    # Limpiar la respuesta de cualquier marcador de código o texto adicional
    cleaned_response = re.sub(r'```json\s*|\s*```', '', response_text)
    cleaned_response = cleaned_response.strip()

    try:
        formatted_text = json.loads(cleaned_response)
        return formatted_text
    except json.JSONDecodeError as e:
        return {
            "error": f"Error al procesar la respuesta: {e}",
            "respuesta_original": response_text,
            "respuesta_limpia": cleaned_response
        }

@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"