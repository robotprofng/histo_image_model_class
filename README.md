# Histology Image Classifier

**Bringing machine learning into oral pathology through an interactive histology image classification application.**

## Project Overview

Developed for an oral pathology professor, this project explores the application of machine learning to histopathological image classification. I built a web application that integrates a trained model into an interactive interface, making it possible to test the model on histology images without interacting directly with the underlying code.

The project sits at the intersection of dentistry, machine learning and software development, translating an AI model into an accessible tool for exploring healthcare research applications.

## Live Application

- **[Try the live demo](https://histoimagemodelclass.streamlit.app/)**
- **[Explore the source code](https://github.com/robotprofng/histo_image_model_class)**

## Technical Implementation

- **Python** — application logic.
- **TensorFlow / Keras** — machine learning model integration.
- **Streamlit** — interactive web application.
- **Model deployment** — making the trained model accessible through a browser-based interface.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/robotprofng/histo_image_model_class.git
cd histo_image_model_class
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies and launch the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Clinical Considerations

This project is an experimental demonstration of machine learning applied to histology image classification. Its clinical performance has not been established, and the application is not intended for clinical diagnosis or to replace professional histopathological assessment.

---

**Healthcare AI · Deep Learning · Python · Application Development**
