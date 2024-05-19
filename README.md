# NLP PDF Reader

## Video Demo

<video width="100%" controls>
  <source src="how.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Prerequisites

Before you start, make sure you have the following:

- An API key for the Gemini API from [AI Studio](https://aistudio.google.com). You can generate the API key by signing up for an account and accessing the Gemini API section.

## Setup

### Backend Setup

1. Create a `.env` file in the root directory of the project.
2. Inside the `.env` file => GOOGLE_API_KEY="your api key"
3. Install the required Python dependencies by running: 
   pip install -r requirements.txt
4. Run the backend server using the following command:
   uvicorn main:app --reload

This will start the backend server, allowing the frontend to communicate with the Gemini API.

### Frontend Setup

1. Navigate to the `front` directory: cd front
2. Install the required npm packages by running:
   npm install axios react-router-dom
3. Start the frontend server: npm start


This command will start the development server for the frontend, allowing you to access the Gemini chat interface in your browser.

## Usage

Once both the backend and frontend servers are running, you can access the Gemini chat interface by navigating to `http://localhost:3000` in your web browser.










