document.addEventListener('DOMContentLoaded', () => {
    loadProfile();

    const profileForm = document.getElementById('profile-form');
    profileForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        updateProfile();
    });
});

/**
 * Helper function to get a cookie by name (if not already in app.js, or for standalone use)
 * This function is identical to the one in app.js. 
 * Consider moving to a shared utility file if more JS files need it.
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

async function loadProfile() {
    try {
        const response = await fetch('/api/user/profile');
        if (response.ok) {
            const profile = await response.json();
            document.getElementById('username-display').textContent = profile.username;
            document.getElementById('username').value = profile.username;
            document.getElementById('email').value = profile.email;
            document.getElementById('data_sharing_consent').checked = profile.data_sharing_consent;
            // Pre-select science interest tags
            setScienceInterestTags(profile.science_interests || []);
        } else {
            document.getElementById('profile-message').textContent = 'Failed to load profile.';
        }
    } catch (error) {
        console.error('Error loading profile:', error);
        document.getElementById('profile-message').textContent = 'Error loading profile.';
    }
}

async function updateProfile() {
    const email = document.getElementById('email').value;
    const dataSharingConsent = document.getElementById('data_sharing_consent').checked;
    // Collect selected tags
    const selectedTags = Array.from(document.querySelectorAll('#scienceInterestsTags input[type=checkbox]:checked')).map(cb => cb.value);
    const scienceInterests = selectedTags;

    const payload = {
        email: email,
        data_sharing_consent: dataSharingConsent,
        science_interests: scienceInterests
    };

    try {
        const csrfToken = getCookie('csrftoken'); // Get CSRF token
        const response = await fetch('/api/user/profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken // Add CSRF token to headers
            },
            body: JSON.stringify(payload),
        });

        const messageDiv = document.getElementById('profile-message');
        if (response.ok) {
            const result = await response.json();
            messageDiv.textContent = result.message || 'Profile updated successfully!';
            messageDiv.className = 'success-message';
            loadProfile(); // Refresh profile details
        } else {
            const errorData = await response.json();
            messageDiv.textContent = `Error updating profile: ${errorData.detail || response.statusText}`;
            messageDiv.className = 'error-message';
        }
    } catch (error) {
        console.error('Error updating profile:', error);
        document.getElementById('profile-message').textContent = 'Error updating profile.';
        document.getElementById('profile-message').className = 'error-message';
    }
} 