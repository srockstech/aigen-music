# S.Rocks.Music AI Generation API

An AI-powered music generation API using Meta's MusicGen model. Generate custom music from text descriptions!

## Features

- Generate music from text descriptions
- Control generation parameters (duration, style, etc.)
- Download generated audio files
- List and manage generated music
- RESTful API with OpenAPI documentation

## API Endpoints

### 1. Health Check
```http
GET /
```
Check if the API is running and get system information.

**Response**
```json
{
    "status": "healthy",
    "model": "musicgen-small",
    "device": "cpu",
    "version": "1.0.0"
}
```

### 2. Generate Music
```http
POST /generate/
```
Generate music from a text description.

**Request Body**
```json
{
    "text": "A lofi hip hop beat with smooth jazz piano and rain sounds",
    "duration": 15,
    "guidance_scale": 3.0,
    "temperature": 1.0
}
```

**Parameters**
- `text` (required): Description of the desired music
- `duration` (optional): Length in seconds (1-30, default: 10)
- `guidance_scale` (optional): Text adherence (0-10, default: 3.0)
- `temperature` (optional): Randomness (0-2, default: 1.0)

**Response**
```json
{
    "file_url": "/files/gen_20250528_134211_a1b2c3d4.wav",
    "duration": 15,
    "timestamp": "20250528_134211",
    "prompt": "A lofi hip hop beat with smooth jazz piano and rain sounds"
}
```

### 3. List Generated Files
```http
GET /files/?limit=50&skip=0
```
Get a list of all generated audio files.

**Parameters**
- `limit` (optional): Maximum files to return (1-100, default: 50)
- `skip` (optional): Number of files to skip (default: 0)

**Response**
```json
[
    {
        "filename": "gen_20250528_134211_a1b2c3d4.wav",
        "url": "/files/gen_20250528_134211_a1b2c3d4.wav",
        "created_at": "2025-05-28 13:42:11",
        "size_kb": 1024.5
    }
]
```

### 4. Download File
```http
GET /files/{filename}
```
Download a specific audio file.

## Example Usage

### Using cURL
```bash
# Generate music
curl -X POST "https://your-api-url/generate/" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Epic orchestral music with dramatic strings and powerful drums",
           "duration": 20
         }'

# List files
curl "https://your-api-url/files/?limit=10"

# Download file
curl -O "https://your-api-url/files/gen_20250528_134211_a1b2c3d4.wav"
```

### Using Python
```python
import requests

# Generate music
response = requests.post(
    "https://your-api-url/generate/",
    json={
        "text": "Ambient electronic music with synth pads",
        "duration": 15
    }
)
result = response.json()
print(f"Generated file: {result['file_url']}")

# Download the file
audio_url = f"https://your-api-url{result['file_url']}"
audio_data = requests.get(audio_url).content
with open("generated_music.wav", "wb") as f:
    f.write(audio_data)
```

## Tips for Better Results

1. **Be Specific in Descriptions**
   - Include instruments
   - Mention genre and mood
   - Describe tempo and style

2. **Example Prompts**
   - "A lofi hip hop beat with smooth jazz piano and rain sounds"
   - "Epic orchestral music with dramatic strings and powerful drums"
   - "Ambient electronic music with synth pads and gentle beats"
   - "Traditional Indian classical music with sitar and tabla"

3. **Optimal Parameters**
   - Start with shorter durations (10-15 seconds)
   - Use guidance_scale=3.0 for balanced results
   - Experiment with temperature for variety

## Technical Details

- Built with FastAPI and PyTorch
- Uses Meta's MusicGen small model
- Runs on CPU (optimized for deployment)
- Automatic file cleanup (keeps last 100 files)

## Rate Limits and Usage

- Maximum duration: 30 seconds
- Maximum stored files: 100
- File format: WAV
- Sample rate: 32000 Hz

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (file doesn't exist)
- 500: Server Error (generation failed)

## Development and Deployment

See [deployment instructions](https://github.com/srockstech/aigen-music#deployment) for details on running your own instance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 