# Deployment Guide for Heart Disease Predictor

## 🚀 Quick Deployment Options

### Option 1: Streamlit Cloud (Easiest - Recommended)

**Step 1: Prepare Repository**
1. Create a GitHub repository
2. Upload all your project files:
   - `app.py`
   - `requirements.txt`
   - `heart_disease_model.pkl`
   - `scaler.pkl`
   - `features.pkl`
   - `.streamlit/config.toml`

**Step 2: Deploy**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - Main file path: `app.py`
   - Python version: 3.9
6. Click "Deploy"

**Access**: Your app will be available at `https://yourname-reponame-streamlit-app.streamlit.app`

---

### Option 2: Heroku

**Step 1: Create Additional Files**

Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

Create `runtime.txt`:
```
python-3.9.16
```

**Step 2: Deploy**
```bash
# Install Heroku CLI first
heroku login
heroku create your-heart-disease-app
git add .
git commit -m "Deploy heart disease predictor"
git push heroku main
```

---

### Option 3: Railway

**Step 1: Setup**
1. Connect your GitHub repository to Railway
2. Railway will auto-detect Python and install requirements

**Step 2: Configure**
Add these environment variables in Railway:
- `PORT`: 8501
- `PYTHONPATH`: /app

**Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 4: Google Cloud Platform (Cloud Run)

**Step 1: Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

**Step 2: Deploy**
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/heart-disease-app

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/heart-disease-app --platform managed
```

---

### Option 5: AWS EC2

**Step 1: Launch EC2 Instance**
- Choose Ubuntu 20.04 LTS
- Configure security group to allow HTTP (port 80) and HTTPS (port 443)

**Step 2: Setup on EC2**
```bash
# Connect to your instance
sudo apt update
sudo apt install python3-pip nginx

# Clone your repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Install dependencies
pip3 install -r requirements.txt

# Run with nohup to keep it running
nohup streamlit run app.py --server.port=8501 &
```

**Step 3: Configure Nginx (Optional)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔧 Pre-Deployment Checklist

### Required Files:
- [ ] `app.py` - Main application
- [ ] `requirements.txt` - Dependencies
- [ ] `heart_disease_model.pkl` - Trained model
- [ ] `scaler.pkl` - Feature scaler
- [ ] `features.pkl` - Feature names
- [ ] `README.md` - Documentation

### Optional Files:
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `Procfile` - For Heroku
- [ ] `Dockerfile` - For containerized deployment
- [ ] `.gitignore` - Git ignore file

### Security & Performance:
- [ ] Remove any hardcoded secrets
- [ ] Add error handling for missing files
- [ ] Test with different input combinations
- [ ] Optimize model file sizes if needed
- [ ] Add rate limiting if necessary

---

## 🌍 Domain Configuration

### Free Domain Options:
1. **Streamlit Cloud**: `yourapp.streamlit.app`
2. **Heroku**: `yourapp.herokuapp.com`
3. **Railway**: `yourapp.railway.app`

### Custom Domain:
1. Purchase domain from provider (Namecheap, GoDaddy, etc.)
2. Configure DNS CNAME record to point to your deployment
3. Update platform settings to use custom domain

---

## 📊 Monitoring & Analytics

### Streamlit Cloud:
- Built-in analytics dashboard
- Error tracking
- Performance metrics

### Custom Analytics:
Add to your `app.py`:
```python
# Google Analytics (add to app.py head)
st.markdown("""
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", unsafe_allow_html=True)
```

---

## 🚨 Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
# Solution: Ensure all packages in requirements.txt
pip freeze > requirements.txt
```

**2. Model File Not Found**
```python
# Add to app.py
if not os.path.exists('heart_disease_model.pkl'):
    st.error("Model file not found. Please upload model files.")
```

**3. Memory Issues**
- Use smaller model files
- Implement model caching
- Consider loading model on-demand

**4. Slow Loading**
```python
# Add caching to app.py
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')
```

---

## 📱 Mobile Optimization

Add to your CSS in `app.py`:
```css
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    .sidebar .sidebar-content {
        width: 100%;
    }
}
```

---

## 🔐 Security Best Practices

1. **Input Validation**: Validate all user inputs
2. **Rate Limiting**: Prevent abuse
3. **HTTPS**: Always use HTTPS in production
4. **Error Handling**: Don't expose internal errors
5. **Logging**: Log important events (without sensitive data)

---

## 📈 Scaling Considerations

### For High Traffic:
1. **Load Balancing**: Use multiple instances
2. **CDN**: Cache static assets
3. **Database**: Store predictions for analytics
4. **API**: Convert to REST API for better performance
5. **Caching**: Cache model predictions

---

Choose the deployment option that best fits your needs:
- **Streamlit Cloud**: Best for quick demos and prototypes
- **Heroku**: Good balance of features and ease
- **Railway**: Modern alternative to Heroku
- **Cloud Providers**: Best for production applications
- **Self-hosted**: Maximum control and customization
