
# 📜 MP4 Audio Transcription API

Este repositório contém uma API REST que processa arquivos de vídeo no formato MP4, extrai o áudio utilizando FFmpeg e realiza a transcrição do áudio com o modelo Whisper. A API também gera um prompt formatado para facilitar a criação de atas de reuniões profissionais.



## 🎯 Funcionalidades

- **Upload de Arquivo MP4**: Permite o envio de arquivos MP4 para processamento.
- **Extração de Áudio**: Extrai o áudio do vídeo utilizando o FFmpeg.
- **Transcrição de Áudio**: Transcreve o áudio extraído usando o modelo Whisper.
- **Geração de Prompt**: Cria um prompt estruturado para formatação de atas de reuniões.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

Certifique-se de que os seguintes itens estão instalados no seu ambiente:

- Python 3.8 ou superior
- FFmpeg instalado no sistema
- Pip (gerenciador de pacotes do Python)
- Ambiente virtual (opcional, mas recomendado)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seuusuario/MP4-Audio-Transcription-API.git
cd MP4-Audio-Transcription-API
```

2. Crie um ambiente virtual e ative:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Certifique-se de que o FFmpeg está configurado corretamente no PATH do sistema.

### Execução

Para iniciar o servidor Flask, execute:

```bash
python app.py
```

O servidor será executado em [http://localhost:5000](http://localhost:5000).

---

## 📋 Endpoints da API

### **POST** `/upload`

Faz o upload de um arquivo MP4, extrai o áudio e retorna a transcrição com um prompt formatado.

#### Parâmetros

| Parâmetro | Tipo   | Descrição                          |
|-----------|--------|------------------------------------|
| `file`    | `file` | Arquivo de vídeo no formato MP4.   |

#### Respostas

| Código | Descrição                                                                 |
|--------|---------------------------------------------------------------------------|
| `200`  | Transcrição e prompt gerados com sucesso.                                |
| `400`  | Erro no envio do arquivo (exemplo: nenhum arquivo enviado ou inválido).  |
| `500`  | Erro interno no processamento.                                           |

#### Exemplo de Resposta (200)

```json
{
  "transcription": "Esta é a transcrição do áudio extraído.",
  "prompt": "Você é um assistente especializado na criação de atas de reuniões..."
}
```

---

## 🛠️ Estrutura do Projeto

```plaintext
.
├── app.py                 # Código principal da aplicação
├── uploads/               # Diretório para upload de arquivos
├── outputs/               # Diretório para saída de arquivos processados
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

---

## 📦 Dependências

As principais dependências utilizadas neste projeto são:

- **Flask**: Framework web para construção da API.
- **Flasgger**: Para documentação automática dos endpoints.
- **FFmpeg**: Ferramenta para processamento de áudio e vídeo.
- **Whisper**: Modelo de transcrição de áudio da OpenAI.

---

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo `LICENSE` para mais informações.

---

## 💡 Observações

- O modelo Whisper utilizado é o `base`, mas pode ser ajustado para outros modelos disponíveis.
- A transcrição está configurada para o idioma **português**, podendo ser alterada conforme necessário.

---

## 🖥️ Contribuindo

Sinta-se à vontade para abrir issues ou pull requests caso encontre problemas ou tenha sugestões para melhorias. 😊

---