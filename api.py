from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger
import os
import ffmpeg
import whisper

# Configurações
UPLOAD_FOLDER = './uploads'
AUDIO_OUTPUT_PATH = './outputs/audio.wav'
WHISPER_MODEL = "base"  # Modelo do Whisper
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
swagger = Swagger(app)

# Funções auxiliares
def extract_audio(video_path, audio_output_path):
    """Extrai o áudio do arquivo MP4 usando FFmpeg."""
    ffmpeg.input(video_path).output(audio_output_path, acodec='pcm_s16le', ar='16000').run(overwrite_output=True)
    return audio_output_path

def transcribe_audio(audio_path):
    """Transcreve o áudio usando o Whisper."""
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, language="portuguese")
    return result["text"]

# Redirecionar a rota principal para o Swagger
@app.route('/')
def index():
    return redirect(url_for('flasgger.apidocs'))

# Rotas da API
@app.route('/upload', methods=['POST'])
def upload_and_transcribe():
    """
    Faz o upload de um arquivo MP4, extrai o áudio e retorna a transcrição com um prompt formatado.
    ---
    tags:
      - Transcrição de Áudio
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: Arquivo de vídeo no formato MP4.
    responses:
      200:
        description: Transcrição e prompt gerados com sucesso.
        schema:
          type: object
          properties:
            transcription:
              type: string
              description: Transcrição do áudio.
            prompt:
              type: string
              description: Prompt gerado a partir da transcrição.
      400:
        description: Erro no envio do arquivo.
        schema:
          type: object
          properties:
            error:
              type: string
              description: Mensagem de erro.
      500:
        description: Erro interno no processamento.
        schema:
          type: object
          properties:
            error:
              type: string
              description: Mensagem de erro.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400
    
    # Salvar o arquivo enviado
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    try:
        # Extrair áudio
        if not os.path.exists('./outputs'):
            os.makedirs('./outputs')
        audio_path = extract_audio(video_path, AUDIO_OUTPUT_PATH)

        # Transcrever áudio
        transcription = transcribe_audio(audio_path)

        # Criar prompt para retorno
        prompt = f"""
        Você é um assistente especializado na criação de atas de reuniões profissionais. Abaixo está a transcrição de uma reunião. 
        Por favor, formate-a em uma ata bem estruturada com os seguintes elementos:

        1. **Título da Reunião**: Insira um título claro e breve.
        2. **Data e Hora**: Baseado no contexto ou use 'Data não especificada' se a data não for fornecida.
        3. **Participantes**: Liste os nomes mencionados ou adicione 'Participantes não especificados' se não forem identificados.
        4. **Resumo**: Forneça um resumo conciso dos principais tópicos discutidos.
        5. **Pontos Principais**:
            - Enumere os principais pontos discutidos durante a reunião.
            - Destaque decisões tomadas e ações atribuídas, se aplicável.
        6. **Próximos Passos**:
            - Liste tarefas ou ações sugeridas para os participantes.
            - Inclua prazos se houver menção.
        7. **Encerramento**: Adicione uma nota final agradecendo a participação de todos.

        ### Transcrição da reunião:
        {transcription}

        Certifique-se de que o formato final esteja limpo e bem organizado para ser enviado por e-mail.
        """
        
        return jsonify({"transcription": transcription, "prompt": prompt})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Inicialização da API
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
