document.addEventListener('DOMContentLoaded', () => {
    // Add a small delay to allow the page to fully load and cookies to be available
    setTimeout(() => {
        loadUserProfile();
    }, 100);
    // Remove the old loadTargets() call since targets are now loaded via the sidebar
    // The targets page template will call updateTargetList() directly
});

async function loadUserProfile() {
    try {
        console.log('Loading user profile from targets.js...');
        const response = await fetch('/api/user/profile', {
            credentials: 'include' // Ensure cookies are sent
        });
        
        console.log('User profile response status:', response.status);
        
        if (response.ok) {
            const profile = await response.json();
            console.log('User profile loaded successfully:', profile);
            // Update username if element exists (may not exist on targets page)
            const usernameElement = document.getElementById('username');
            if (usernameElement) {
                usernameElement.textContent = profile.username;
            }
        } else if (response.status === 401) {
            console.warn('User not authenticated, but not redirecting from targets.js');
            // Don't redirect or throw error - let the page handle this gracefully
        } else {
            console.error('Failed to load user profile:', response.status, response.statusText);
        }
    } catch (error) {
        console.error('Error loading user profile:', error);
        // Don't throw the error - let the page continue to function
    }
}

// Remove the old loadTargets function since we're using the sidebar approach
// The updateTargetList function from app.js will handle loading targets into the sidebar

async function removeTarget(ztfid) {
    if (!confirm(`Are you sure you want to remove target ${ztfid}?`)) {
        return;
    }
    try {
        const response = await fetch('/api/remove-target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ztfid: ztfid }),
        });
        if (response.ok) {
            showToast('Target removed successfully!', 'success');
            // Use updateTargetList from app.js instead of loadTargets
            if (typeof updateTargetList === 'function') {
                updateTargetList();
            }
        } else {
            const errorData = await response.json();
            showToast(`Error removing target: ${errorData.detail || response.statusText}`, 'error');
        }
    } catch (error) {
        showToast(`Error removing target: ${error.message}`, 'error');
        console.error('Error removing target:', error);
    }
}

// Basic toast notification function (can be expanded or use a library)
// function showToast(message, type = 'info', duration = 3000) { // Commented out, use global from app.js
//     const container = document.querySelector('.toast-container') || createToastContainer();
//    
//     const toast = document.createElement('div');
//     toast.className = `toast ${type}`;
//     toast.textContent = message;
//    
//     container.appendChild(toast);
//    
//     setTimeout(() => {
//         toast.remove();
//     }, duration);
// }
//
// function createToastContainer() { // Commented out, use global from app.js
//     const container = document.createElement('div');
//     container.className = 'toast-container';
//     document.body.appendChild(container);
//     return container;
// } 