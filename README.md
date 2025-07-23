# 💰 SalaryBoost AI - Intelligent Salary Prediction System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.1.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A sophisticated machine learning web application that predicts salary categories (≤$50K or >$50K) based on demographic and employment features using the Adult Census Income dataset.

## 🌟 Features

- **AI-Powered Predictions**: Advanced Gradient Boosting machine learning model
- **Interactive Web Interface**: Modern, responsive web application with real-time predictions
- **Production Ready**: Deployed on Microsoft Azure Web App Service
- **Comprehensive Data Processing**: Handles categorical encoding, feature scaling, and data validation
- **Visual Analytics**: Beautiful charts and probability displays
- **RESTful API**: JSON-based prediction endpoint for integration with other applications

## 🚀 Live Demo

Experience the application live: [SalaryBoost AI - Azure Deployment](https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/)

🌐 **Deployed on Microsoft Azure** - Central India region for optimal performance

## 📊 Dataset

This project uses the **Adult Census Income Dataset**:
- **Source**: Edunet Foundation educational program
- **Records**: ~32,000 individuals
- **Features**: 14 attributes including age, education, occupation, etc.
- **Target**: Binary classification (≤50K vs >50K annual income)

### Key Features Used:
- **Demographics**: Age, Gender, Race, Native Country
- **Education**: Education Level, Education Years
- **Employment**: Work Class, Occupation, Hours per Week
- **Financial**: Capital Gain, Capital Loss
- **Family**: Marital Status, Relationship

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask 3.1.1**: Lightweight web framework
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation and analysis
- **Joblib**: Model serialization
- **Gunicorn**: Production WSGI server

### Frontend
- **HTML5 & CSS3**: Modern web standards with responsive design
- **JavaScript**: Interactive form handling and API calls
- **Chart.js**: Prediction probability visualization
- **CSS Animations**: Dynamic background and UI effects

### Machine Learning
- **Gradient Boosting Classifier**: Primary prediction model
- **Label Encoding**: Categorical feature processing
- **Standard Scaling**: Feature normalization

## 📁 Project Structure

```
salary-prediction-ml/
│
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── salary_prediction_model.ipynb   # Jupyter notebook with ML development
├── startup.txt                     # Deployment configuration
│
├── data/
│   └── adult 3.csv                 # Training dataset
│
├── models/
│   ├── best_gradient_boosting_model.pkl  # Trained ML model
│   ├── label_encoder.pkl                 # Target variable encoder
│   └── scaler.pkl                        # Feature scaler
│
├── templates/
│   └── index.html                  # Web interface template
│
├── PPT_Resource/                   # Presentation materials
│   ├── algorithm_comparison.png
│   ├── code_screenshot_*.png
│   ├── outlier_before.png
│   ├── outlier_after.png
│   └── summary.png
│
└── myenv/                          # Virtual environment
    ├── Scripts/
    ├── Lib/
    └── pyvenv.cfg
```

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shaileshukla529/salary-prediction-ml.git
   cd salary-prediction-ml
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   
   # Windows
   myenv\Scripts\activate
   
   # macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🔧 Usage

### Web Interface
1. Navigate to the [live application](https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/)
2. Fill in the prediction form with personal details:
   - Age, Education Level, Occupation
   - Work Class, Marital Status, Relationship
   - Gender, Race, Native Country
   - Work Hours, Capital Gain/Loss
3. Click "Predict Salary" to get results
4. View prediction with confidence probability

### API Usage

**Base URL**: `https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net`

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "age": 35,
  "workclass": "Private",
  "education": "Bachelors",
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "gender": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

**Response**:
```json
{
  "prediction": ">50K",
  "probability": [0.25, 0.75],
  "success": true
}
```

## 🎯 Model Performance

Our Gradient Boosting model achieves:
- **Accuracy**: ~85%
- **Precision**: ~82%
- **Recall**: ~79%
- **F1-Score**: ~80%

*Detailed performance metrics and comparison with other algorithms can be found in the Jupyter notebook.*

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

#### Using Startup Configuration
```bash
gunicorn --bind=0.0.0.0 --timeout 600 app:app
```

#### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "--bind=0.0.0.0:5000", "--timeout", "600", "app:app"]
```

#### Cloud Platform Deployment

**🔵 Microsoft Azure (Current Deployment)**
- **Live URL**: https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/
- **Region**: Central India
- **Service**: Azure Web App Service
- **Deployment**: GitHub Actions CI/CD via Azure Deployment Center
- **Configuration**: Uses `startup.txt` with gunicorn settings
- **Scaling**: Auto-scaling enabled for production workloads

**Other Cloud Platforms**
- **AWS**: Deploy with Elastic Beanstalk or Lambda

## 🔍 API Endpoints

**Base URL**: `https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net`

| Endpoint | Method | Description | Example URL |
|----------|--------|-------------|-------------|
| `/` | GET | Main web interface | [Open App](https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/) |
| `/predict` | POST | Make salary predictions | POST to /predict |
| `/health` | GET | Health check for monitoring | [Health Check](https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/health) |
| `/test` | GET | Simple test endpoint | [Test Endpoint](https://salary-predictor-ml-app-fqfae7b2asgndyg0.centralindia-01.azurewebsites.net/test) |

## 🛡️ Error Handling

The application includes comprehensive error handling:
- Input validation and sanitization
- Model loading verification
- Graceful fallbacks for missing data
- Detailed error logging and user feedback

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📈 Future Enhancements

- [ ] **Multi-class Classification**: Predict specific salary ranges
- [ ] **Real-time Data**: Integration with live job market data
- [ ] **Advanced Models**: Deep learning and ensemble methods
- [ ] **Mobile App**: Native mobile application
- [ ] **A/B Testing**: Model performance comparison
- [ ] **User Accounts**: Save and track predictions
- [ ] **Data Visualization**: Advanced analytics dashboard

## 📚 Resources

- [Jupyter Notebook](salary_prediction_model.ipynb) - Complete ML development process
- [Flask Documentation](https://flask.palletsprojects.com/) - Web framework guide
- [Scikit-learn Documentation](https://scikit-learn.org/) - ML library reference
- [Azure Web App Service](https://docs.microsoft.com/en-us/azure/app-service/) - Deployment platform

## 🏆 Acknowledgments

- **Edunet Foundation** for the Adult dataset and educational support
- **Flask Community** for the excellent web framework
- **Scikit-learn Contributors** for the machine learning tools
- **Microsoft Azure** for reliable cloud hosting platform

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shailesh Shukla**
- GitHub: [@Shaileshukla529](https://github.com/Shaileshukla529)
- Email: [shaileshukla529@gmail.com](mailto:shaileshukla529@gmail.com)
- LinkedIn: [Shailesh Shukla](https://www.linkedin.com/in/shailesh-shukla-540789309)

## 📞 Support

If you have any questions or need help with the project:

- **Issues**: [GitHub Issues](https://github.com/Shaileshukla529/salary-prediction-ml/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Shaileshukla529/salary-prediction-ml/discussions)
- **Email**: [shaileshukla529@gmail.com](mailto:shaileshukla529@gmail.com)

---

⭐ **Star this repository if you found it helpful!** ⭐

*Made with ❤️ and ☕ in Python*
