# VTON Frontend

A simple, elegant frontend for testing the Virtual Try-On GPU inference server.

## Features

- 🎨 **Premium Dark Theme** - Modern glassmorphism design with smooth animations
- 📤 **Drag & Drop Upload** - Easy image upload for person and garment images
- ⚙️ **Configurable Parameters** - Adjust seed, steps, and CFG scale
- 📊 **Real-time Status** - Server connection and model status indicators
- 💾 **Download Results** - Save generated images locally
- 📱 **Responsive Design** - Works on desktop and mobile

## Quick Start

1. **Start the frontend server:**
   ```bash
   cd vton_frontend
   python serve.py
   ```
   
   Or use any static file server:
   ```bash
   npx serve .
   # or
   python -m http.server 3000
   ```

2. **Open in browser:**
   ```
   http://localhost:3000
   ```

3. **Configure the GPU server URL** (default: `http://localhost:8080`)

4. **Upload images and run inference!**

## Configuration

The frontend stores configuration in localStorage:
- `gpu_server_url` - GPU server URL
- `gpu_server_token` - Authentication token (if required)

## Files

```
vton_frontend/
├── index.html      # Main HTML structure
├── styles.css      # Premium CSS styling
├── app.js          # Application logic
├── serve.py        # Simple Python server
└── README.md       # This file
```

## API Integration

The frontend uses the `/infer` endpoint of the GPU server for synchronous inference:

```
POST /infer
Content-Type: multipart/form-data

Form Fields:
- masked_user_image: Person image with green mask
- garment_image: Garment to try on
- seed: Random seed (default: 42)
- steps: Inference steps (default: 4)
- cfg: CFG scale (default: 1.0)

Response:
- 200: PNG image data
- 429: GPU busy
- 503: Models not loaded
```

## Development

To modify the frontend:

1. Edit the files directly - no build step required
2. Refresh the browser to see changes
3. Use browser DevTools for debugging

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+
