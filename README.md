
# SmartScribe

### AI-Based Voice-Enabled Exam System for Visually Impaired Students

SmartScribe is an AI-powered examination system that uses face verification, voice verification, speech recognition and text-to-speech to provide an accessible examination experience for visually impaired students.

## Technologies Used

- Python
- Flask
- HTML, CSS, JavaScript
- SQLite
- Face Recognition
- Voice Recognition
- Speech Recognition
- Text-to-Speech
- OpenCV
- Librosa
- ReportLab

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
````

Run the application:

```bash
python app.py
```

Open the URL shown in the terminal in Google Chrome.

---

# Instructions for Evaluators

The complete project can be tested using the following steps.

### 1. Admin Login

Open the **Admin Login** page.

Use the demo admin credentials:

**Username:** `admin`
**Password:** `admin123`

### 2. Register a Student

After logging in:

* Go to **Create/Register Student**
* Enter a student name
* Enter a registration number
* Upload/take a clear face photo
* Record a clear voice sample
* Click **Create Student**

> Evaluators can use their own photo and voice for this demonstration.

### 3. Create an Exam

Go to **Create Exam**.

* Enter an exam name
* Set the duration
* Upload the question paper PDF
* Create the exam

### 4. Logout

Logout from the Admin Portal.

### 5. Student Verification

Open the **Student Login** page.

Enter the same registration number used during registration.

The system will perform:

**Face Verification → Voice Verification**

Use the same person's face and voice that were registered.

### 6. Take the Exam

After successful verification:

* Select the available exam
* Questions will be read aloud
* Answer the questions using your voice
* Use voice commands such as **NEXT** and **REPEAT**

### 7. Submit

After answering the questions, say/use **SUBMIT**.

The system will generate the student's **answer-sheet PDF**.

---

## Expected Output

The evaluator should be able to verify:

* Admin login
* Student registration
* Face registration and verification
* Voice registration and verification
* Exam creation
* Voice-based question answering
* Exam submission
* Automatic answer-sheet PDF generation

## Note

For the best experience, use **Google Chrome** and allow **Camera** and **Microphone** permissions when requested.

