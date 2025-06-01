# Transient Recommender Server

Server component for the Transient Recommender system that handles user authentication and label synchronization.

## Features

- User authentication with JWT tokens and API keys
- Secure storage of user preferences and labels
- Data sharing consent management
- PostgreSQL database backend
- FastAPI-based REST API

## Installation

1. Create a Python virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL database and create a new database:
```sql
CREATE DATABASE transient_recommender;
```

4. Create a `.env` file with the following configuration:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/transient_recommender
SECRET_KEY=your-secret-key-here
# For CSRF Protection - generate a strong random key, e.g., using: openssl rand -hex 32
CSRF_SECRET_KEY=your-csrf-secret-key-here 
```

5. Initialize the database:
```bash
alembic upgrade head
```

## Running the Server

Start the server with:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API documentation will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Usage Example

Here's how to use the remote sync functionality in your transient recommender client:

```python
from transient_recommender import TransientRecommender

# Initialize with remote sync enabled
recommender = TransientRecommender(
    username="your_username",
    enable_remote_sync=True,
    remote_url="https://your-server.com"
)

# First-time setup: register and consent to data sharing
recommender.remote_sync.register(
    email="your@email.com",
    password="your_password",
    data_sharing_consent=True
)

# Or login if already registered
recommender.remote_sync.login(password="your_password")

# Get recommendations (will now automatically sync with remote server)
recommendations = recommender.get_recommendations()
```

Note: By enabling remote sync, you agree to share your transient labels with the community. This helps improve recommendations for everyone. You can revoke this consent at any time through the `update_consent` method.

## API Endpoints

The API provides endpoints for user management, recommendations, object interactions, and more. 
For detailed, interactive documentation, please see:
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

Below is a summary of the main API endpoints:

### Authentication & User Management
- `POST /register`: Register a new user (expects form data).
- `POST /token`: Login to get an OAuth2 access token (expects form data).
- `POST /login`: Handles direct form submission for login and sets an access token cookie.
- `GET /logout`: Logs out the current user and clears the access token cookie.
- `GET /api/user/profile`: Retrieves the profile information for the currently authenticated user.
- `POST /api/user/profile`: Updates the profile information (email, data sharing consent, science interests) for the currently authenticated user.

### Recommendations & Voting
- `GET /api/recommendations`: Fetches a list of transient recommendations for the user. 
    - Query parameters: `science_case` (string), `count` (integer).
- `POST /api/vote`: Submits a user's vote (like, dislike, target) for a specific transient. Can also include tags and notes in the metadata.
- `POST /api/skip`: Marks a transient as skipped by the user.

### Targets
- `GET /api/targets`: Retrieves the list of objects targeted by the user.
- `POST /api/remove-target`: Removes an object from the user's target list (effectively changing its status from 'target' to 'like').

### Tags & Notes (Interactions)
- `GET /api/tags/{ztfid}`: Gets all tags applied by the user to a specific ZTF object.
- `POST /api/tags/{ztfid}`: Saves or updates the list of tags for a specific ZTF object by the user.
- `GET /api/notes/{ztfid}`: Retrieves any notes written by the user for a specific ZTF object.
- `POST /api/notes/{ztfid}`: Saves or updates the notes for a specific ZTF object by the user.

### Finder Charts
- `POST /api/generate-finders`: Initiates the generation of finder charts for all objects currently in the user's target list.

### Statistics
- `GET /api/stats`: Provides statistics for the current user, such as counts of likes, dislikes, targets, etc.

### Admin
- `POST /api/update-feature-bank`: (Admin Only) Triggers an update of the system's feature bank from its source CSV file.

## Security Measures

1. Password hashing using bcrypt
2. JWT tokens for authentication
3. API keys for automated access
4. Data sharing consent tracking
5. HTTPS required in production
6. Rate limiting
7. Input validation
8. CSRF Protection via Double Submit Cookie method (using starlette-csrf)

## Production Deployment

For production deployment:

1. Use a production-grade WSGI server (e.g., Gunicorn)
2. Set up HTTPS using a reverse proxy (e.g., Nginx)
3. Use environment variables for all sensitive configuration
4. Set up proper database backup procedures
5. Configure logging
6. Set up monitoring

Example production startup command:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 