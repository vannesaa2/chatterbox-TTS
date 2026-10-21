import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando el modelo...")

modelo= ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto="Hola, bienvenidos a Inteligencia Artificial Generativa, vamos a crear imagenes, texto, audio y si el computador aguanta video."

print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7,#Emoción
    cfg_weight=0.4, #Velocidad y ritmo
    temperature=0.8,#Controla la variabilidad
    repetition_penalty=1.2, #Penaliza repeticiones
    min_p= 0.05, #Filra opciones improbables
    top_p=1.0 #Controla el muestro probabilistico
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio generado ")
