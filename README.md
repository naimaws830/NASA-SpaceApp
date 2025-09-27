📚 Complete Local Setup Guide
🔧 Step 1: Install Prerequisites
Before starting, install these tools on your computer:

1. Install Python 3.8+
Download from: https://www.python.org/downloads/
During installation, ☑️ check "Add Python to PATH"
Verify: Open Command Prompt/Terminal → python --version

2. Install Node.js 16+
Download from: https://nodejs.org/
Choose LTS version
Verify: node --version and npm --version

4. Install Yarn
npm install -g yarn
Verify: yarn --version

4. Install VS Code Extensions (Recommended)
Python
ES7+ React/Redux/React-Native snippets
Tailwind CSS IntelliSense
Prettier - Code formatter

🗂️ Step 2: Project Setup
1. Download and Extract Project

# Download the GitHub zip file
# Extract to your desired location, e.g., C:\projects\ or ~/projects/
# You should have a folder like: climate-dashboard/
2. Open in VS Code

# Open VS Code
# File → Open Folder → Select your climate-dashboard folder
# Or use terminal: code climate-dashboard/
🐍 Step 3: Backend Setup (Python/FastAPI)
1. Open VS Code Terminal

View → Terminal (or Ctrl+ / Cmd+)
2. Navigate to Backend

cd backend
3. Create Virtual Environment

# Create virtual environment
python -m venv venv

# Activate it
# Windows (Command Prompt):
venv\Scripts\activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# macOS/Linux:
source venv/bin/activate

# You should see (venv) prefix in your terminal
4. Create Clean Requirements File Create backend/requirements.txt:

fastapi==0.110.1
uvicorn[standard]==0.25.0
motor==3.3.1
pymongo==4.5.0
python-dotenv==1.1.1
pydantic==2.11.9
numpy==2.3.3
aiohttp==3.12.15
netCDF4==1.7.2
python-multipart==0.0.20
requests==2.32.5
scipy==1.16.2
5. Install Dependencies

pip install -r requirements.txt

⚛️ Step 4: Frontend Setup (React/Node.js)
1. Open New Terminal Tab
Terminal → New Terminal (or Ctrl+Shift+ / Cmd+Shift+)

2. Navigate to Frontend
cd frontend

3. Install Dependencies
yarn install

4. Create Environment File Create frontend/.env:
REACT_APP_BACKEND_URL=http://localhost:8000

5. Verify Configuration Files Exist Check these files exist in frontend/:

package.json
craco.config.js
tailwind.config.js
postcss.config.js

🗄️ Step 5: Database Setup (Local MongoDB)

# macOS with Homebrew:
brew install mongodb-community
brew services start mongodb-community

# Windows: Download MongoDB Community Server
# https://www.mongodb.com/try/download/community

# Ubuntu/Linux:
sudo apt install mongodb
sudo systemctl start mongodb

🚀 Step 7: Running the Application

1. Start Backend (Terminal 1)
cd backend
# Make sure virtual environment is activated (you should see (venv))
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
You should see:

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.

2. Start Frontend (Terminal 2)
cd frontend
yarn start
Your browser should open automatically to http://localhost:3000
