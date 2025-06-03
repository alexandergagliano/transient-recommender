/**
 * Transient Recommender Web Application
 */

// Global variables
let currentRecommendations = [];
let currentIndex = 0;
let currentTags = new Map(); // Store tags for each object
let currentNotes = new Map(); // Store notes for each object
let targetList = new Set(); // Keep track of targets
let toastTimeouts = []; // Keep track of toast timeouts

// Audio Recording Variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let currentSessionId = null; // To store a simple session ID

const SCIENCE_CASES_FOR_TAGS = ["long-lived", "anomalous", "snia-like", "ccsn-like", "precursor"];

let voteCountsRefreshInterval = null;
let currentScienceCase = 'snia-like'; // Store current science case

// Mobile touch gesture variables
let touchStartX = 0;
let touchStartY = 0;
let touchEndX = 0;
let touchEndY = 0;
let isSwiping = false;

/**
 * Helper function to get a cookie by name
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

/**
 * Initialize on DOM content loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    
    // Apply saved preferences
    applyPreferencesOnLoad();
    
    // Set up theme switcher
    setupThemeToggle();
    
    // Add keyboard shortcuts hint
    addKeyboardShortcutsHint();
    
    // Initial loading of recommendations - only if on recommendations page
    if (document.getElementById('current-object-container')) {
        loadRecommendations();
    }
    updateTargetList();

    // Set up handlers for science case change
    const scienceSelect = document.getElementById('scienceSelect');
    if (scienceSelect) {
        // Initialize current science case from the page
        currentScienceCase = scienceSelect.value || 'snia-like';
        
        scienceSelect.addEventListener('change', (e) => {
            currentScienceCase = e.target.value;
            updateRecommendations();
        });
    }

    // Set up handler for start_ztfid input
    const startZtfidInput = document.getElementById('startZtfidInput');
    if (startZtfidInput) {
        startZtfidInput.addEventListener('change', updateRecommendations);
    }

    // Set up handlers for observing constraints
    const applyObsConstraintsButton = document.getElementById('applyObsConstraints');
    if (applyObsConstraintsButton) {
        applyObsConstraintsButton.addEventListener('click', updateRecommendations);
    }

    // Set up keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        // Only apply to recommendations page
        if (!document.getElementById('current-object-container')) return; 

        // Toggle theme - check for Ctrl+D specifically and prevent other actions
        if (e.key.toLowerCase() === 'd' && e.ctrlKey) {
            e.preventDefault();
            toggleTheme();
            return; // Exit early to prevent dislike action
        }

        // Navigation
        if (e.key === 'ArrowLeft') showPrevious();
        else if (e.key === 'ArrowRight') showNext();
        
        // Voting - only trigger if Ctrl is NOT pressed
        else if ((e.key === '1' || e.key === 'l') && !e.ctrlKey) handleVote(getCurrentObject(), 'like');
        else if ((e.key === '2' || e.key === 'd') && !e.ctrlKey) handleVote(getCurrentObject(), 'dislike');
        else if ((e.key === '3' || e.key === 't') && !e.ctrlKey) handleVote(getCurrentObject(), 'target');
        else if ((e.key === '4' || e.key === 's') && !e.ctrlKey) handleNext(getCurrentObject());
        
        // Notes - focus on comment textarea
        else if (e.key === 'n' && !e.ctrlKey) {
            const commentTextarea = document.getElementById('newCommentText');
            if (commentTextarea) {
                commentTextarea.focus();
            }
        }
    });

    // Set up tag buttons
    document.querySelectorAll('.tag-btn').forEach(btn => {
        btn.addEventListener('click', handleTagClick);
    });
    
    // Set up custom tag inputs for each category
    document.querySelectorAll('.add-custom-tag-btn').forEach(btn => {
        btn.addEventListener('click', handleAddCustomTagByCategory);
    });

    document.querySelectorAll('.custom-tag-input').forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const category = e.target.dataset.category;
                const btn = e.target.parentElement.querySelector('.add-custom-tag-btn');
                if (btn) btn.click();
            }
        });
    });

    // Set up spectra button
    // const spectraButton = document.getElementById('spectraBtn');
    // if (spectraButton) {
    //     spectraButton.addEventListener('click', loadSpectra);
    // }
    
    // Set up toast container
    setupToastContainer();

    // DISABLED: Polling was causing unwanted page refreshes while users were working
    // startPollingRecommendations();

    // Vote buttons
    const likeButton = document.getElementById('likeBtn');
    const dislikeButton = document.getElementById('dislikeBtn');
    const targetButton = document.getElementById('targetBtn');
    const skipButton = document.getElementById('skipBtn');
    
    if (likeButton) likeButton.addEventListener('click', () => handleVote(getCurrentObject(), 'like'));
    
    if (dislikeButton) dislikeButton.addEventListener('click', () => handleVote(getCurrentObject(), 'dislike'));
    
    if (targetButton) targetButton.addEventListener('click', () => handleVote(getCurrentObject(), 'target'));
    
    if (skipButton) skipButton.addEventListener('click', () => handleNext(getCurrentObject()));

    const saveNoteButton = document.getElementById('saveNoteButton');
    if (saveNoteButton) saveNoteButton.addEventListener('click', saveNotes);

    const addCommentButton = document.getElementById('addCommentButton');
    if (addCommentButton) addCommentButton.addEventListener('click', addComment);

    const addTagButton = document.getElementById('addTagButton');
    if (addTagButton) addTagButton.addEventListener('click', handleAddCustomTag);

    // Add Enter key support for custom tag input
    const customTagInput = document.getElementById('customTagInput');
    if (customTagInput) {
        customTagInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleAddCustomTag();
            }
        });
    }

    // Add event listeners for predefined tag buttons
    const tagButtonsContainer = document.getElementById('tagButtons');
    if (tagButtonsContainer) {
        tagButtonsContainer.querySelectorAll('.tag-btn').forEach(btn => {
            btn.addEventListener('click', handleTagClick);
        });
    }

    // Event listener for the new record audio button
    const recordAudioButton = document.getElementById('recordAudioButton');
    if (recordAudioButton) {
        recordAudioButton.addEventListener('click', toggleAudioRecording);
    }

    populateScienceCaseTagButtons();

    // Load extraction status on page load - only if extraction status elements exist
    if (document.getElementById('extractionStatus')) {
        loadExtractionStatus();
    }

    // Set up handler for trigger extraction button
    const triggerExtractionButton = document.getElementById('extract-features-button');
    if (triggerExtractionButton) {
        triggerExtractionButton.addEventListener('click', triggerFeatureExtraction);
        console.log('Extract features button event listener attached');
    } else {
        console.warn('Extract features button not found');
    }

    // Set up mobile touch gestures
    setupMobileGestures();
    
    // Show mobile tutorial if needed
    showMobileTutorialIfNeeded();

    // Setup target button functionality
    setupTargetModal();

        // Check if demo should be shown (instant for new users)
    setTimeout(() => {
    checkDemoAvailability();
    }, 50); // Very short delay to let page finish loading
    
    // Request notification permissions for feature extraction alerts
    requestNotificationPermission();
    
    // Set up preset buttons for lookback days
    setupLookbackPresetButtons();
    
    // Set up collapsible card headers
    setupCollapsibleHeaders();
    
    // Setup clear constraints button
    const clearConstraintsBtn = document.getElementById('clearConstraints');
    if (clearConstraintsBtn) {
        clearConstraintsBtn.addEventListener('click', clearConstraints);
    }
    
    // Setup realtime toggle
    const realtimeToggle = document.getElementById('realtimeToggle');
    if (realtimeToggle) {
        realtimeToggle.addEventListener('change', toggleRealtimeMode);
        console.log('Realtime toggle event listener attached');
    }
    
    // Make toggleRealtimeMode globally available
    window.toggleRealtimeMode = toggleRealtimeMode;
});

function applyPreferencesOnLoad() {
    const preferences = JSON.parse(localStorage.getItem('userPreferences') || '{}');
    
    // Apply theme
    const savedTheme = localStorage.getItem('theme') || preferences.theme || 'light';
    console.log('Applying theme on page load:', savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
    console.log('Theme applied, data-theme attribute:', document.documentElement.getAttribute('data-theme'));
    
    // Apply default science case
    const scienceSelect = document.getElementById('scienceSelect');
    if (scienceSelect && preferences.defaultScienceCase) {
        scienceSelect.value = preferences.defaultScienceCase;
        currentScienceCase = preferences.defaultScienceCase;
    }
    
    // Apply default telescope
    const telescopeSelect = document.getElementById('obsConstraintTelescope');
    if (telescopeSelect && preferences.defaultTelescope) {
        telescopeSelect.value = preferences.defaultTelescope;
    }
    
    // Apply default magnitude limit
    const magLimitInput = document.getElementById('obsConstraintMagLimit');
    if (magLimitInput && preferences.defaultMagLimit) {
        magLimitInput.value = preferences.defaultMagLimit;
    }
    
    // Apply default observing days
    const obsDaysInput = document.getElementById('obsConstraintDays');
    if (obsDaysInput && preferences.defaultObsDays) {
        obsDaysInput.value = preferences.defaultObsDays;
    }
}

/**
 * Set up theme toggle functionality
 */
function setupThemeToggle() {
    console.log('setupThemeToggle called');
    
    // Create theme toggle if it doesn't exist
    if (!document.querySelector('.theme-switch-wrapper')) {
        console.log('Creating theme toggle');
        // Target the second navbar-nav (the one with login/logout links)
        const navbarNavs = document.querySelectorAll('.navbar-nav');
        const rightNavbar = navbarNavs.length > 1 ? navbarNavs[1] : navbarNavs[0]; // Use second one if available, fallback to first
        console.log('Found navbar-nav elements:', navbarNavs.length, 'Using:', rightNavbar ? 'found' : 'not found');
        
        if (rightNavbar) {
            const themeToggle = document.createElement('li');
            themeToggle.className = 'nav-item theme-switch-wrapper';
            themeToggle.innerHTML = `
                <div class="theme-toggle-tabs">
                    <button class="theme-tab active" data-theme="light">
                        <i class="bi bi-sun"></i> Light
                    </button>
                    <button class="theme-tab" data-theme="dark">
                        <i class="bi bi-moon"></i> Dark
                    </button>
                </div>
            `;
            
            // Insert the theme toggle before the last nav item (logout/login)
            if (rightNavbar.children.length > 0) {
                rightNavbar.insertBefore(themeToggle, rightNavbar.lastElementChild);
            } else {
                rightNavbar.appendChild(themeToggle);
            }
            console.log('Theme toggle created and added to navbar');
            
            // Add event listeners to theme tabs
            const themeTabs = themeToggle.querySelectorAll('.theme-tab');
            console.log('Found theme tabs:', themeTabs.length);
            themeTabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    console.log('Theme tab clicked:', tab.dataset.theme);
                    setTheme(tab.dataset.theme);
                });
            });
            
            // Set initial state based on saved theme
            const currentTheme = localStorage.getItem('theme') || 'light';
            console.log('Setting initial theme:', currentTheme);
            setTheme(currentTheme);
        } else {
            console.error('navbar-nav not found, cannot add theme toggle');
        }
    } else {
        console.log('Theme toggle already exists');
        
        // Still set up event listeners for existing toggle
        const themeTabs = document.querySelectorAll('.theme-tab');
        console.log('Found existing theme tabs:', themeTabs.length);
        themeTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                console.log('Existing theme tab clicked:', tab.dataset.theme);
                setTheme(tab.dataset.theme);
            });
        });
        
        // Apply current theme
        const currentTheme = localStorage.getItem('theme') || 'light';
        console.log('Applying current theme to existing toggle:', currentTheme);
        setTheme(currentTheme);
    }
}

/**
 * Set theme and update tab states
 */
function setTheme(theme) {
    console.log('setTheme called with:', theme);
    
    // Set the data-theme attribute on the document element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Also set it on the body for redundancy
    document.body.setAttribute('data-theme', theme);
    
    // Save to localStorage
    localStorage.setItem('theme', theme);
    
    console.log('Applied theme attribute:', document.documentElement.getAttribute('data-theme'));
    console.log('Body theme attribute:', document.body.getAttribute('data-theme'));
    
    // Update tab states
    const themeTabs = document.querySelectorAll('.theme-tab');
    console.log('Updating theme tabs:', themeTabs.length);
    themeTabs.forEach(tab => {
        const isActive = tab.dataset.theme === theme;
        tab.classList.toggle('active', isActive);
        console.log(`Tab ${tab.dataset.theme} active:`, isActive);
    });
    
    // Show confirmation toast
    showToast(`Switched to ${theme} mode`, 'info');
    
    // Force a style recalculation
    document.documentElement.style.cssText = document.documentElement.style.cssText;
}

/**
 * Toggle between light and dark theme (for keyboard shortcut)
 */
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

/**
 * Add keyboard shortcuts hint
 */
function addKeyboardShortcutsHint() {
    const hintElement = document.createElement('div');
    hintElement.className = 'keyboard-shortcuts-hint';
    hintElement.innerHTML = `
        <div>Shortcuts: 
        <span class="keyboard-key">←</span>/<span class="keyboard-key">→</span> Navigate, 
        <span class="keyboard-key">1</span>/<span class="keyboard-key">l</span> Like, 
        <span class="keyboard-key">2</span>/<span class="keyboard-key">d</span> Dislike, 
        <span class="keyboard-key">3</span>/<span class="keyboard-key">t</span> Target, 
        <span class="keyboard-key">4</span>/<span class="keyboard-key">s</span> Skip, 
        <span class="keyboard-key">n</span> Notes, 
        <span class="keyboard-key">Ctrl+D</span> Dark mode</div>
    `;
    document.body.appendChild(hintElement);
}

/**
 * Set up toast container
 */
function setupToastContainer() {
    if (!document.querySelector('.toast-container')) {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
}

/**
 * Show a toast notification
 */
function showToast(message, type = 'info', duration = 3000) {
    // Check if notifications are enabled for this type
    if (typeof window.isNotificationEnabled === 'function') {
        const notificationTypes = {
            'success': 'notifyVotes',
            'error': 'notifyVotes', 
            'info': 'notifyNewRecommendations',
            'warning': 'notifyFeatureExtraction'
        };
        
        const prefType = notificationTypes[type] || 'notifyVotes';
        if (!window.isNotificationEnabled(prefType)) {
            return; // Don't show notification if disabled
        }
    }
    
    // Try multiple selectors for toast container
    let toastContainer = document.querySelector('.toast-container') || document.getElementById('toast-container');
    
    if (!toastContainer) {
        // Create toast container if it doesn't exist
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        toastContainer.id = 'toast-container';
        toastContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.style.cssText = `
        background: ${type === 'success' ? '#d4edda' : type === 'error' ? '#f8d7da' : type === 'warning' ? '#fff3cd' : '#d1ecf1'};
        color: ${type === 'success' ? '#155724' : type === 'error' ? '#721c24' : type === 'warning' ? '#856404' : '#0c5460'};
        border: 1px solid ${type === 'success' ? '#c3e6cb' : type === 'error' ? '#f5c6cb' : type === 'warning' ? '#ffeaa7' : '#bee5eb'};
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        pointer-events: auto;
        animation: slideIn 0.3s ease;
        max-width: 300px;
        word-wrap: break-word;
    `;
    
    const iconMap = {
        success: '✓',
        error: '✗',
        warning: '⚠',
        info: 'ℹ'
    };
    
    const icon = iconMap[type] || iconMap.info;
    
    toast.innerHTML = `
        <div style="display: flex; align-items: flex-start; gap: 8px;">
            <span style="font-weight: bold;">${icon}</span>
            <div style="flex: 1;">${message}</div>
            <button onclick="this.parentElement.parentElement.remove()" 
                    style="background: none; border: none; font-size: 16px; cursor: pointer; padding: 0; margin-left: 8px;">
                ×
            </button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, duration);
}

/**
 * Remove a toast notification
 */
function removeToast(toast) {
    toast.style.animation = 'slideOut 0.3s ease forwards';
    
    // Clear the timeout
    const toastIndex = toastTimeouts.findIndex(t => t.toast === toast);
    if (toastIndex !== -1) {
        clearTimeout(toastTimeouts[toastIndex].timeout);
        toastTimeouts.splice(toastIndex, 1);
    }
    
    // Remove the element after animation
    setTimeout(() => {
        toast.remove();
    }, 300);
}

/**
 * Show loading spinner
 */
function showLoading(show) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Get the current object's ZTFID
 */
function getCurrentObject() {
    console.log('getCurrentObject called, currentIndex:', currentIndex, 'recommendations length:', currentRecommendations.length);
    
    if (currentIndex < 0 || currentIndex >= currentRecommendations.length) {
        console.warn('getCurrentObject: Index out of bounds', currentIndex, 'of', currentRecommendations.length);
        return undefined;
    }
    
    const obj = currentRecommendations[currentIndex];
    if (!obj) {
        console.warn('getCurrentObject: No object at current index', currentIndex);
        return undefined;
    }
    
    // Handle both ZTFID and ztfid properties
    const ztfid = obj.ztfid || obj.ZTFID;
    if (!ztfid) {
        console.warn('getCurrentObject: Object has no ztfid property', obj);
        return undefined;
    }
    
    console.log('getCurrentObject returning:', ztfid);
    return ztfid;
}

/**
 * Load recommendations from the server
 */
async function loadRecommendations() {
    console.log('loadRecommendations called');
    showLoading(true);
    try {
        // Use stored science case or get from select element
        const scienceSelect = document.getElementById('scienceSelect');
        if (scienceSelect) {
            currentScienceCase = scienceSelect.value || currentScienceCase;
        }
        console.log('Current science case:', currentScienceCase);
        
        // Get observing constraints
        const telescope = document.getElementById('obsConstraintTelescope')?.value || '';
        const days = document.getElementById('obsConstraintDays')?.value || '';
        const magLimit = document.getElementById('obsConstraintMagLimit')?.value || '';
        const startZtfid = document.getElementById('startZtfidInput')?.value.trim() || '';
        
        // Get real-time mode parameters
        const realtimeToggle = document.getElementById('realtimeToggle');
        const recentDaysSelect = document.getElementById('recentDaysSelect');
        const isRealtimeMode = realtimeToggle?.checked || false;
        const recentDays = recentDaysSelect?.value || '7';

        let apiUrl = `/api/recommendations?science_case=${currentScienceCase}&count=10`;
        if (telescope) apiUrl += `&obs_telescope=${telescope}`;
        if (days) apiUrl += `&obs_days=${days}`;
        if (magLimit) apiUrl += `&obs_mag_limit=${magLimit}`;
        if (startZtfid) apiUrl += `&start_ztfid=${startZtfid}`;
        if (isRealtimeMode) {
            apiUrl += `&realtime_mode=true&recent_days=${recentDays}`;
            console.log(`Real-time mode: filtering to last ${recentDays} days`);
        }

        console.log('Fetching recommendations from:', apiUrl);
        const response = await fetch(apiUrl);
        console.log('API response status:', response.status);
        
        // Handle the no objects available error (422 status)
        if (response.status === 422) {
            console.log('No objects available (422 response)');
            const errorData = await response.json();
            if (errorData.detail && errorData.detail.error === 'no_objects_available') {
                const errorInfo = errorData.detail;
                let message = errorInfo.message || 'No objects match your current constraints.';
                
                // Add suggestions to the message
                if (errorInfo.suggestions && errorInfo.suggestions.length > 0) {
                    message += '\n\nSuggestions:\n• ' + errorInfo.suggestions.join('\n• ');
                }
                
                // Show a more helpful error message
                showToast(message, 'warning', 8000); // Show longer for readability
                
                // Clear current recommendations
                currentRecommendations = [];
                currentIndex = 0;
                
                // Hide object display
                const objectDisplay = document.querySelector('.object-display');
                if (objectDisplay) {
                    objectDisplay.style.display = 'none';
                }
                
                // Show constraint info in the main area
                const mainContent = document.querySelector('.recommendation-content') || document.querySelector('.container');
                if (mainContent) {
                    const constraintInfo = document.createElement('div');
                    constraintInfo.className = 'no-objects-message alert alert-warning';
                    constraintInfo.innerHTML = `
                        <h4><i class="fas fa-filter"></i> No Objects Match Your Constraints</h4>
                        <p>${errorInfo.message}</p>
                        ${errorInfo.suggestions && errorInfo.suggestions.length > 0 ? `
                            <h6>Suggestions:</h6>
                            <ul>
                                ${errorInfo.suggestions.map(s => `<li>${s}</li>`).join('')}
                            </ul>
                        ` : ''}
                        <button class="btn btn-primary mt-2" onclick="clearConstraints()">
                            <i class="fas fa-times"></i> Clear All Constraints
                        </button>
                    `;
                    
                    // Remove any existing no-objects message
                    const existingMessage = mainContent.querySelector('.no-objects-message');
                    if (existingMessage) {
                        existingMessage.remove();
                    }
                    
                    // Insert at the beginning
                    mainContent.insertBefore(constraintInfo, mainContent.firstChild);
                }
                
                return;
            }
        }
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        currentRecommendations = await response.json();
        console.log('Loaded recommendations:', currentRecommendations.length, 'objects');
        console.log('First object:', currentRecommendations[0]);
        
        currentIndex = 0;
        
        // Remove any existing no-objects message
        const existingMessage = document.querySelector('.no-objects-message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Show object display again
        const objectDisplay = document.querySelector('.object-display');
        if (objectDisplay) {
            objectDisplay.style.display = 'block';
        }
        
        // If no recommendations left, show message
        if (currentRecommendations.length === 0) {
            console.log('No recommendations returned from API');
            showToast('No more unvoted objects available!', 'info');
            return;
        }
        
        console.log('Calling updateObjectDisplay...');
        updateObjectDisplay();
        showToast(`Loaded ${currentRecommendations.length} recommendations`, 'success');
    } catch (error) {
        console.error('Error loading recommendations:', error);
        showToast('Error loading recommendations. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Update recommendations (e.g., after science case change)
 */
function updateRecommendations() {
    loadRecommendations();
}

/**
 * Update the target list sidebar
 */
async function updateTargetList() {
    try {
        const response = await fetch('/api/targets', {
            credentials: 'include' // Ensure cookies are sent
        });
        
        if (response.status === 401) {
            // User not authenticated - handle gracefully without throwing error
            console.warn('User not authenticated, cannot load targets');
            const targetListContainer = document.getElementById('target-list-grid');
            const targetCountElement = document.getElementById('target-count-badge');
            
            if (targetListContainer) {
                targetListContainer.innerHTML = '<p class="text-muted text-center">Please log in to view targets</p>';
            }
            if (targetCountElement) {
                targetCountElement.textContent = '0';
            }
            return;
        }
        
        if (!response.ok) throw new Error(`Error: ${response.status}`);
        
        const targets = await response.json();
        
        // Try both container types (modal and table)
        const targetListContainer = document.getElementById('target-list-grid');
        const modalCountElement = document.getElementById('target-modal-count');
        
        if (!targetListContainer) {
            console.warn('No target list container found');
            return;
        }
        
        targetListContainer.innerHTML = '';
        
        if (targets.length === 0) {
            targetListContainer.innerHTML = '<p class="text-muted text-center">No targets yet</p>';
        } else {
            targets.forEach(target => {
                const targetCard = document.createElement('div');
                targetCard.className = 'target-card';
                targetCard.dataset.ztfid = target.ztfid;
                targetCard.dataset.ra = target.ra;
                targetCard.dataset.dec = target.dec;
                targetCard.innerHTML = `
                    <div class="target-card-header">
                        <a href="https://alerce.online/object/${target.ztfid}" target="_blank" class="target-card-title">${target.ztfid}</a>
                        <button class="target-card-remove" onclick="removeTarget('${target.ztfid}')" title="Remove from targets">
                            ×
                        </button>
                    </div>
                    <div class="target-card-info">
                        <div class="target-card-info-item">
                            <span class="target-card-info-label">RA</span>
                            <span class="target-card-info-value">${target.ra?.toFixed(5) || 'N/A'}</span>
                        </div>
                        <div class="target-card-info-item">
                            <span class="target-card-info-label">Dec</span>
                            <span class="target-card-info-value">${target.dec?.toFixed(5) || 'N/A'}</span>
                        </div>
                        <div class="target-card-info-item">
                            <span class="target-card-info-label">Added</span>
                            <span class="target-card-info-value">${target.created_at ? new Date(target.created_at).toLocaleDateString() : 'N/A'}</span>
                        </div>
                    </div>
                `;
                targetListContainer.appendChild(targetCard);
            });
        }
        
        // Update target counts for all possible count elements
        const countElements = [
            document.getElementById('target-count-badge'),
            modalCountElement,
            document.getElementById('targetCountTargetsPage')
        ].filter(el => el); // Remove null elements
        
        countElements.forEach(el => {
            el.textContent = targets.length;
        });
        
        // Update the targets button state
        const targetsButton = document.getElementById('targets-button');
        if (targetsButton) {
            if (targets.length > 0) {
                targetsButton.disabled = false;
                targetsButton.classList.add('btn-outline-warning');
                targetsButton.classList.remove('btn-outline-light');
            } else {
                targetsButton.disabled = true;
                targetsButton.classList.add('btn-outline-light');
                targetsButton.classList.remove('btn-outline-warning');
            }
        }
        
        console.log(`Updated target list: ${targets.length} targets`);
        
    } catch (error) {
        console.error('Error updating target list:', error);
        // Don't let this error break the page
    }
}

/**
 * Remove a target from the target list
 */
async function removeTarget(ztfid) {
    try {
        console.log('Removing target:', ztfid);
        
        const response = await fetch('/api/remove-target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ztfid: ztfid }),
            credentials: 'include'  // Ensure cookies (JWT token) are included
        });
        
        console.log('📡 Remove target response:', response.status, response.statusText);
        
        if (response.ok) {
            const result = await response.json();
            console.log('Target removal successful:', result);
            showToast(`Removed ${ztfid} from targets`, 'success');
            await updateTargetList();
        } else {
            const errorText = await response.text();
            console.error('Remove target failed:', response.status, errorText);
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }
    } catch (error) {
        console.error('Error removing target:', error);
        showToast(`Failed to remove target: ${error.message}`, 'error', 5000);
    }
}

/**
 * Handle tag button click with category support
 */
async function handleTagClick() {
    const ztfid = getCurrentObject();
    if (!ztfid) return;
    
    const tag = this.dataset.tag;
    const category = this.dataset.category || 'general';
    
    this.classList.toggle('active');
    
    // Get existing tags for the object, organized by category
    let currentObjectTags = currentTags.get(ztfid) || {
        science: new Set(),
        photometry: new Set(),
        host: new Set(),
        general: new Set()
    };

    if (this.classList.contains('active')) {
        currentObjectTags[category].add(tag);
    } else {
        currentObjectTags[category].delete(tag);
    }

    try {
        // Convert Sets to Arrays for API
        const tagsForAPI = {};
        for (const [cat, tagSet] of Object.entries(currentObjectTags)) {
            tagsForAPI[cat] = Array.from(tagSet);
        }

        const csrfToken = getCookie('csrftoken');
        const response = await fetch(`/api/tags/${ztfid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify({ tags: tagsForAPI })
        });
        
        if (response.ok) {
            currentTags.set(ztfid, currentObjectTags);
            displayCurrentObjectTags(ztfid);
            showToast(`Tag '${tag}' ${this.classList.contains('active') ? 'added' : 'removed'}.`, 'success');
        } else {
            // Revert UI change on error
            this.classList.toggle('active'); 
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error saving tags:', error);
        showToast('Error saving tags. Please try again.', 'error');
    }
}

/**
 * Handle adding custom tags by category
 */
async function handleAddCustomTagByCategory() {
    const ztfid = getCurrentObject();
    if (!ztfid) return;
    
    const category = this.dataset.category;
    const input = this.parentElement.querySelector('.custom-tag-input');
    const tagName = input.value.trim();
    
    if (!tagName) {
        showToast('Please enter a tag name.', 'warning');
        return;
    }
    
            // Get existing tags for the object
        let currentObjectTags = currentTags.get(ztfid) || {
            science: new Set(),
            photometry: new Set(),
            host: new Set(),
            general: new Set()
        };

    // Add the new tag to the appropriate category
    currentObjectTags[category].add(tagName);
    
    try {
        // Convert Sets to Arrays for API
        const tagsForAPI = {};
        for (const [cat, tagSet] of Object.entries(currentObjectTags)) {
            tagsForAPI[cat] = Array.from(tagSet);
        }

        const csrfToken = getCookie('csrftoken');
        const response = await fetch(`/api/tags/${ztfid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify({ tags: tagsForAPI })
        });

        if (response.ok) {
            currentTags.set(ztfid, currentObjectTags);
            displayCurrentObjectTags(ztfid);
            input.value = ''; // Clear the input
            showToast(`Custom ${category} tag '${tagName}' added.`, 'success');
        } else {
            // Revert changes on server error
            currentObjectTags[category].delete(tagName);
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error adding custom tag:', error);
        showToast('Error adding custom tag. Please try again.', 'error');
        currentObjectTags[category].delete(tagName);
        currentTags.set(ztfid, currentObjectTags);
    }
}

/**
 * Display current object tags organized by category
 */
function displayCurrentObjectTags(ztfid) {
    const tagsContainer = document.getElementById('currentTags');
    if (!tagsContainer) return;

    const tags = currentTags.get(ztfid);
    if (!tags) {
        tagsContainer.innerHTML = '<p class="text-muted">No tags yet.</p>';
        return;
    }

    let html = '';
    const categoryColors = {
        science: '#007bff',
        spectra: '#28a745', 
        photometry: '#ffc107',
        host: '#17a2b8',
        general: '#6c757d'
    };

    for (const [category, tagSet] of Object.entries(tags)) {
        if (tagSet.size > 0) {
            const categoryName = category.charAt(0).toUpperCase() + category.slice(1);
            html += `
                <div class="current-tags-category">
                    <h6 style="color: ${categoryColors[category]}">${categoryName} Tags:</h6>
                    <div>
                        ${Array.from(tagSet).map(tag => 
                            `<span class="current-tag-item" style="border-left: 3px solid ${categoryColors[category]}">
                                ${escapeHtml(tag)}
                                <button class="btn btn-sm btn-outline-danger tag-delete-btn" 
                                        onclick="deleteTag('${ztfid}', '${escapeHtml(tag)}', '${category}')"
                                        title="Delete tag">
                                    <i class="bi bi-x"></i>
                                </button>
                            </span>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
    }

    if (html === '') {
        html = '<p class="text-muted">No tags yet.</p>';
    }

    tagsContainer.innerHTML = html;

    // Update predefined tag buttons to show active state
    updateTagButtonStates(ztfid);
}

/**
 * Delete a specific tag for an object
 */
async function deleteTag(ztfid, tagName, category) {
    if (!confirm(`Are you sure you want to delete the tag "${tagName}"?`)) {
        return;
    }
    
    try {
        // Get current tags for the object
        let currentObjectTags = currentTags.get(ztfid) || {
            science: new Set(),
            spectra: new Set(),
            photometry: new Set(),
            host: new Set(),
            general: new Set()
        };

        // Remove the tag from the appropriate category
        currentObjectTags[category].delete(tagName);
        
        // Convert Sets to Arrays for API
        const tagsForAPI = {};
        for (const [cat, tagSet] of Object.entries(currentObjectTags)) {
            tagsForAPI[cat] = Array.from(tagSet);
        }

        const csrfToken = getCookie('csrftoken');
        const response = await fetch(`/api/tags/${ztfid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify({ tags: tagsForAPI })
        });

        if (response.ok) {
            currentTags.set(ztfid, currentObjectTags);
            displayCurrentObjectTags(ztfid);
            showToast(`Tag '${tagName}' deleted.`, 'success');
        } else {
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error deleting tag:', error);
        showToast('Error deleting tag. Please try again.', 'error');
    }
}

/**
 * Update predefined tag button states
 */
function updateTagButtonStates(ztfid) {
    const tags = currentTags.get(ztfid);
    if (!tags) return;

    document.querySelectorAll('.tag-btn').forEach(btn => {
        const tag = btn.dataset.tag;
        const category = btn.dataset.category || 'general';
        
        if (tags[category] && tags[category].has(tag)) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

/**
 * Load and display spectra for the current object
 * DISABLED: Spectra functionality removed from frontend
 */
/*
async function loadSpectra(ztfid) {
    if (!ztfid) return;

    const section = document.getElementById('spectraSection');
    const loadingDiv = document.getElementById('spectraLoading');
    const contentDiv = document.getElementById('spectraContent');
    const noSpectraDiv = document.getElementById('noSpectraMessage');

    if (!section || !contentDiv) return;

    // Show section and loading state
    section.style.display = 'block';
    loadingDiv.style.display = 'block';
    contentDiv.innerHTML = '';
    noSpectraDiv.style.display = 'none';

    try {
        const response = await fetch(`/api/spectra/${ztfid}`);
        
        loadingDiv.style.display = 'none';
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        if (data.spectra && data.spectra.length > 0) {
            // Display spectra in compact format for sidebar
            let html = '';

            data.spectra.forEach((spectrum, index) => {
                html += `
                    <div class="spectrum-item mb-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <div class="fw-bold text-primary small">${escapeHtml(spectrum.name)}</div>
                                <div class="text-muted small">
                                    ${escapeHtml(spectrum.date)} • ${escapeHtml(spectrum.instrument)}
                                </div>
                                <div class="text-muted small">
                                    <i class="fas fa-telescope"></i> ${escapeHtml(spectrum.telescope)}
                                </div>
                            </div>
                            ${spectrum.download_link ? `
                                <a href="${spectrum.download_link}" target="_blank" 
                                   class="btn btn-sm btn-outline-primary ms-2" title="Download spectrum">
                                    <i class="fas fa-download"></i>
                                </a>
                            ` : ''}
                        </div>
                    </div>
                `;
            });

            html += `
                <div class="text-center mt-2">
                    <small class="text-muted">
                        ${data.total_count} spectrum${data.total_count !== 1 ? 'a' : ''} found
                        <a href="${data.source_url}" target="_blank" class="ms-2">
                            <i class="fas fa-external-link-alt"></i> View on WiseREP
                        </a>
                    </small>
                </div>
            `;

            contentDiv.innerHTML = html;
        } else {
            // No spectra found
            noSpectraDiv.style.display = 'block';
            
            // Add WiseREP link for manual search
            if (data.source_url) {
                noSpectraDiv.innerHTML = `
                    <i class="fas fa-search"></i> No spectra found<br>
                    <a href="${data.source_url}" target="_blank" class="small">
                        <i class="fas fa-external-link-alt"></i> Search WiseREP manually
                    </a>
                `;
            }
        }

    } catch (error) {
        console.error('Error loading spectra:', error);
        loadingDiv.style.display = 'none';
        
        // Show error state
        contentDiv.innerHTML = `
            <div class="alert alert-warning alert-sm p-2">
                <small><strong>Error loading spectra</strong><br>
                ${escapeHtml(error.message)}</small>
            </div>
        `;
        
        // Still show WiseREP link for manual search
        const wiserepUrl = `https://www.wiserep.org/search/spectra?ztfname=${ztfid}`;
        contentDiv.innerHTML += `
            <div class="text-center mt-2">
                <a href="${wiserepUrl}" target="_blank" class="small">
                    <i class="fas fa-external-link-alt"></i> Search WiseREP manually
                </a>
            </div>
        `;
    }
}
*/

/**
 * Load object metadata with updated categorized tags format
 */
async function loadObjectMetadata(ztfid) {
    if (!ztfid) return;
    
    try {
        const [tagsResponse, notesResponse] = await Promise.all([
            fetch(`/api/tags/${ztfid}`),
            fetch(`/api/notes/${ztfid}`)
        ]);
        
        if (tagsResponse.ok) {
            const tagsData = await tagsResponse.json();
            
            // Convert arrays back to Sets for easier manipulation
            const tagsWithSets = {};
            for (const [category, tagArray] of Object.entries(tagsData)) {
                // Skip spectra category (removed from frontend)
                if (category !== 'spectra') {
                    tagsWithSets[category] = new Set(tagArray);
                }
            }
            
            currentTags.set(ztfid, tagsWithSets);
            displayCurrentObjectTags(ztfid);
        }
        
        if (notesResponse.ok) {
            const notes = await notesResponse.json();
            const notesElement = document.getElementById('objectNotes');
            if (notesElement) {
                notesElement.value = '';
            }
            currentNotes.set(ztfid, notes.text || '');
            displaySavedNotes(ztfid);
        }
        
        // Load comments for the new comment system
        await loadComments(ztfid);
    } catch (error) {
        console.error('Error loading metadata:', error);
    }
}

/**
 * Save notes for the current object
 */
async function saveNotes() {
    const ztfid = getCurrentObject();
    if (!ztfid) return;
    
    const notesTextarea = document.getElementById('objectNotes');
    const newNote = notesTextarea.value.trim();
    
    if (!newNote) {
        showToast('Please enter a note before saving.', 'warning');
        return;
    }
    
    try {
        // Get existing notes
        const existingNotes = currentNotes.get(ztfid) || '';
        
        // Create timestamp
        const now = new Date();
        const timestamp = now.toLocaleString();
        
        // Append new note with timestamp
        const noteEntry = `[${timestamp}] ${newNote}`;
        const updatedNotes = existingNotes ? `${existingNotes}\n\n${noteEntry}` : noteEntry;
        
        const csrfToken = getCookie('csrftoken');
        const response = await fetch(`/api/notes/${ztfid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify({ text: updatedNotes })
        });
        
        if (response.ok) {
            currentNotes.set(ztfid, updatedNotes);
            
            // Clear the textarea
            notesTextarea.value = '';
            
            // Update the display
            displaySavedNotes(ztfid);
            
            showToast('Note saved successfully!', 'success');
        } else {
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error saving notes:', error);
        showToast('Error saving notes. Please try again.', 'error');
    }
}

/**
 * Display saved notes in a comment-like format
 */
function displaySavedNotes(ztfid) {
    const savedNotesContainer = document.getElementById('savedNotes');
    if (!savedNotesContainer) return;
    
    const notes = currentNotes.get(ztfid) || '';
    
    if (!notes.trim()) {
        savedNotesContainer.innerHTML = '<p class="text-muted small">No notes yet.</p>';
        return;
    }
    
    // Split notes by double newlines to separate individual note entries
    const noteEntries = notes.split('\n\n').filter(entry => entry.trim());
    
    let notesHtml = '<div class="saved-notes-list">';
    
    noteEntries.forEach((entry, index) => {
        // Try to extract timestamp and content
        const timestampMatch = entry.match(/^\[([^\]]+)\]\s*(.*)$/s);
        
        if (timestampMatch) {
            const timestamp = timestampMatch[1];
            const content = timestampMatch[2].trim();
            
            notesHtml += `
                <div class="note-entry mb-2 p-2 border rounded bg-light">
                    <div class="note-content">${content.replace(/\n/g, '<br>')}</div>
                    <div class="note-timestamp text-muted small mt-1">
                        <i class="bi bi-clock"></i> ${timestamp}
                    </div>
                </div>
            `;
        } else {
            // Legacy note without timestamp
            notesHtml += `
                <div class="note-entry mb-2 p-2 border rounded bg-light">
                    <div class="note-content">${entry.replace(/\n/g, '<br>')}</div>
                    <div class="note-timestamp text-muted small mt-1">
                        <i class="bi bi-clock"></i> Legacy note
                    </div>
                </div>
            `;
        }
    });
    
    notesHtml += '</div>';
    savedNotesContainer.innerHTML = notesHtml;
}

/**
 * Update tag button states to match the current object's tags
 */
function updateTagButtons(tags) {
    // This function is now effectively replaced by displayCurrentObjectTags
    // And direct updates within handleTagClick
    // console.warn("updateTagButtons is deprecated, use displayCurrentObjectTags")
    const ztfid = getCurrentObject();
    if (ztfid) {
        displayCurrentObjectTags(ztfid);
    }
}

/**
 * Update the display to show the current object
 */
function updateObjectDisplay() {
    console.log('updateObjectDisplay called, currentRecommendations:', currentRecommendations.length, 'currentIndex:', currentIndex);
    
    if (currentRecommendations.length === 0) {
        console.log('No recommendations available');
        document.getElementById('current-object-container').style.display = 'none';
        return;
    }

    const object = currentRecommendations[currentIndex];
    console.log('Current object:', object);
    
    if (!object) {
        console.error('Object is undefined at index', currentIndex);
        return;
    }
    
    // Handle both ZTFID and ztfid properties
    const ztfid = object.ztfid || object.ZTFID;
    if (!ztfid) {
        console.error('Object has no ztfid property:', object);
        return;
    }
    
    console.log('Processing object with ztfid:', ztfid);
    
    const container = document.getElementById('current-object-container');
    const objectId = document.getElementById('currentObjectId');
    const objectDetails = document.getElementById('currentObjectDetails');
    const mainViewer = document.getElementById('mainViewer');

    if (!container || !objectId || !objectDetails || !mainViewer) {
        console.error('Required DOM elements not found');
        return;
    }

    // Update object info
    objectId.textContent = ztfid;
    objectDetails.textContent = `RA: ${object.ra?.toFixed(4) || 'N/A'}, Dec: ${object.dec?.toFixed(4) || 'N/A'}`;

    // Update iframe with ALeRCE link (with CORS-friendly attributes)
    mainViewer.innerHTML = `
        <iframe src="https://alerce.online/object/${ztfid}" 
                style="width: 100%; height: 100%; border: none;"
                referrerpolicy="no-referrer-when-downgrade"
                allow="web-share"
                loading="eager">
        </iframe>
    `;

    // Show container
    container.style.display = 'block';

    // Load object-specific data with defensive checks
    console.log('Loading object-specific data for:', ztfid);
    loadObjectMetadata(ztfid);
    displayCurrentObjectTags(ztfid);
    loadComments(ztfid);
    loadAudioRecordings(ztfid);
    loadClassifierBadges(ztfid);
    loadSlackVotes(ztfid);     
    loadVoteCounts(ztfid);

    // Update URL without page reload
    if (window.history && window.history.pushState) {
        const url = new URL(window.location);
        url.searchParams.set('ztfid', ztfid);
        window.history.pushState({}, '', url);
    }

    // Start vote counts refresh for this object
    startVoteCountsRefresh();
}

/**
 * Show the next object in the list
 */
function showNext() {
    if (currentIndex < currentRecommendations.length - 1) {
        currentIndex++;
        updateObjectDisplay();
    }
}

/**
 * Show the previous object in the list
 */
function showPrevious() {
    if (currentIndex > 0) {
        currentIndex--;
        updateObjectDisplay();
    }
}

/**
 * Handle voting (like/dislike/target)
 */
async function handleVote(ztfid, voteType) {
    if (!ztfid) return;
    
    try {
        // Show loading animation
        showLoading(true);
        
        // Get the current object's data
        const obj = currentRecommendations[currentIndex];
        
        // Get current tags for this object (now organized by category)
        const tagsData = currentTags.get(ztfid) || {
            science: new Set(),
            photometry: new Set(),
            host: new Set(),
            general: new Set()
        };
        
        // Convert tags to arrays for API (flatten all categories into a single array for backward compatibility)
        const allTags = [];
        for (const [category, tagSet] of Object.entries(tagsData)) {
            allTags.push(...Array.from(tagSet));
        }
        
        // Get current notes for this object
        const notes = currentNotes.get(ztfid) || '';
        
        // Prepare the complete data payload
        const payload = {
            ztfid: ztfid,
            vote: voteType,
            science_case: document.getElementById('scienceSelect').value,
            metadata: {
                tags: allTags, // Flattened tags for backward compatibility
                categorized_tags: tagsData, // New categorized format
                notes: notes,
                object_details: {
                    ra: obj.ra,
                    dec: obj.dec,
                    latest_magnitude: obj.latest_magnitude
                }
            }
        };

        const csrfToken = getCookie('csrftoken');
        const response = await fetch('/api/vote', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify(payload)
        });
        
        if (response.ok) {
            // Update vote counts to show the new vote
            await loadVoteCounts(ztfid);
            
            // If this was a target vote, update the target list
            if (voteType === 'target') {
                await updateTargetList();
                showToast(`Added ${ztfid} to targets`, 'success');
            } else {
                // Use appropriate emoji for each vote type
                let voteMessage = '';
                if (voteType === 'like') voteMessage = `Liked ${ztfid}`;
                else if (voteType === 'dislike') voteMessage = `Disliked ${ztfid}`;
                
                showToast(voteMessage, 'success');
            }
            
            // STAY on the current page - do NOT advance to next object
            // Users can review their vote, add tags/notes, then click "Next" to advance
            
        } else {
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error voting:', error);
        showToast('Error saving vote. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Handle advancing to next object (formerly skip)
 */
async function handleNext(ztfid) {
    if (!ztfid) return;
    
    try {
        showLoading(true);
        
        const csrfToken = getCookie('csrftoken');
        const response = await fetch('/api/skip', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-csrftoken': csrfToken
            },
            body: JSON.stringify({ ztfid: ztfid })
        });
        
        if (response.ok) {
            showToast(`Moved to next object`, 'info');
            
            // Remove this object from recommendations and advance
            currentRecommendations.splice(currentIndex, 1);
            
            // If we've removed all recommendations, reload them
            if (currentRecommendations.length === 0) {
                await loadRecommendations();
            } else {
                // If we're at the end of the list, go back one
                if (currentIndex >= currentRecommendations.length) {
                    currentIndex = currentRecommendations.length - 1;
                }
                updateObjectDisplay();
            }
        } else {
            throw new Error(`Error: ${response.status}`);
        }
    } catch (error) {
        console.error('Error advancing to next object:', error);
        showToast('Error advancing to next object. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Generate finder charts for all targets and download as zip
 */
async function generateFinderCharts() {
    try {
        showLoading(true);
        showToast('Generating finder charts...', 'info', 3000);
        
        const response = await fetch('/api/generate-finders', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            // Check if response is a file (zip) or JSON error
            const contentType = response.headers.get('content-type');
            
            if (contentType && contentType.includes('application/zip')) {
                // It's a zip file - trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                // Extract filename from response headers or use default
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'finder_charts.zip';
                if (contentDisposition) {
                    const matches = contentDisposition.match(/filename="(.+)"/);
                    if (matches && matches[1]) {
                        filename = matches[1];
                    }
                }
                
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                showToast('Finder charts downloaded successfully!', 'success', 5000);
            } else {
                // It's a JSON response (likely an error)
                const result = await response.json();
                
                if (result.status === 'failed') {
                    let message = `Finder chart generation failed for all ${result.total_targets} targets`;
                    if (result.failed_objects && result.failed_objects.length > 0) {
                        message += `\n\nErrors: ${result.failed_objects.slice(0, 2).join(', ')}`;
                        if (result.failed_objects.length > 2) {
                            message += ` and ${result.failed_objects.length - 2} more...`;
                        }
                    }
                    showToast(message, 'error', 10000);
                } else {
                    // Fallback for any other status
                    showToast(result.message || 'Finder chart generation completed', 'info', 8000);
                }
            }
        } else {
            const errorText = await response.text();
            throw new Error(`Error: ${response.status} - ${errorText}`);
        }
    } catch (error) {
        console.error('Error generating finder charts:', error);
        showToast('Error generating finder charts. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

/**
 * Convert decimal to HMS (Hours:Minutes:Seconds) format
 */
function decimalToHMS(decimal) {
    let sign = decimal >= 0 ? 1 : -1;
    decimal = Math.abs(decimal);
    
    // Convert to hours (divide by 15 for RA)
    let totalHours = decimal / 15;
    
    // Extract hours
    let hours = Math.floor(totalHours);
    
    // Convert remaining hours to minutes
    let totalMinutes = (totalHours - hours) * 60;
    let minutes = Math.floor(totalMinutes);
    
    // Convert remaining minutes to seconds
    let seconds = ((totalMinutes - minutes) * 60).toFixed(2);
    
    // Ensure two digits for minutes and seconds
    hours = hours.toString().padStart(2, '0');
    minutes = minutes.toString().padStart(2, '0');
    seconds = parseFloat(seconds).toFixed(2).padStart(5, '0');
    
    return `${hours}:${minutes}:${seconds}`;
}

/**
 * Convert decimal to DMS (Degrees:Minutes:Seconds) format
 */
function decimalToDMS(decimal) {
    let sign = decimal >= 0 ? '+' : '-';
    decimal = Math.abs(decimal);
    
    // Extract degrees
    let degrees = Math.floor(decimal);
    
    // Convert remaining degrees to minutes
    let totalMinutes = (decimal - degrees) * 60;
    let minutes = Math.floor(totalMinutes);
    
    // Convert remaining minutes to seconds
    let seconds = ((totalMinutes - minutes) * 60).toFixed(2);
    
    // Ensure two digits for minutes and seconds
    degrees = degrees.toString().padStart(2, '0');
    minutes = minutes.toString().padStart(2, '0');
    seconds = parseFloat(seconds).toFixed(2).padStart(5, '0');
    
    return `${sign}${degrees}:${minutes}:${seconds}`;
}

/**
 * Save target list as a text file
 */
function saveTargetList() {
    // Create the content for the file
    let content = '';
    
    // Add header
    content += '# Target List\n';
    content += '# Name                     RA (HH:MM:SS.ss)    Dec (DD:MM:SS.ss)\n';
    content += '#------------------------------------------------------------\n';
    
    // Add each target with fixed-width formatting
    // Look for both old .target-item and new .target-card elements
    const targetElements = [
        ...document.querySelectorAll('.target-item'),
        ...document.querySelectorAll('.target-card')
    ];
    
    targetElements.forEach(el => {
        const ztfid = el.dataset.ztfid;
        const ra = parseFloat(el.dataset.ra);
        const dec = parseFloat(el.dataset.dec);
        
        if (ztfid && !isNaN(ra) && !isNaN(dec)) {
            // Convert coordinates
            const raHMS = decimalToHMS(ra);
            const decDMS = decimalToDMS(dec);
            
            // Format with fixed width
            content += `${ztfid.padEnd(25)} ${raHMS.padEnd(18)} ${decDMS}\n`;
        }
    });
    
    if (content.split('\n').length <= 4) {
        showToast('No targets to download', 'warning');
        return;
    }
    
    // Create blob and download
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'target_list.txt';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    showToast('Target list downloaded successfully', 'success');
}

// Polling for new recommendations
let pollingIntervalId = null;
const POLLING_INTERVAL = 30000; // Poll every 30 seconds

function startPollingRecommendations() {
    if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
    }
    // Don't poll if on a page other than recommendations
    if (!document.getElementById('current-object-container')) { // Check for new container ID
        return;
    }
    
    // DISABLED: Aggressive polling was causing unwanted page refreshes and science case changes
    // Users reported that the page was periodically refreshing and changing transients/science cases
    // This polling was too disruptive to the user experience
    
    /*
    pollingIntervalId = setInterval(async () => {
        // Check if there are new recommendations without disrupting the user
        // This is a simplified check. A more robust solution might involve a dedicated endpoint 
        // or checking a timestamp of the latest available recommendation.
        try {
            const scienceSelect = document.getElementById('scienceSelect');
            const scienceCase = scienceSelect ? scienceSelect.value : 'snia-like'; // Default or selected
            
            // Get observing constraints for polling check
            const telescope = document.getElementById('obsConstraintTelescope')?.value || '';
            const days = document.getElementById('obsConstraintDays')?.value || '';
            const magLimit = document.getElementById('obsConstraintMagLimit')?.value || '';
            const startZtfid = document.getElementById('startZtfidInput')?.value.trim() || '';

            let pollUrl = `/api/recommendations?science_case=${scienceCase}&count=1`;
            if (telescope) pollUrl += `&obs_telescope=${telescope}`;
            if (days) pollUrl += `&obs_days=${days}`;
            if (magLimit) pollUrl += `&obs_mag_limit=${magLimit}`;
            if (startZtfid) pollUrl += `&start_ztfid=${startZtfid}`;

            const response = await fetch(pollUrl);
            if (response.ok) {
                const newRecs = await response.json();
                if (newRecs && newRecs.length > 0) {
                    // Check if the new recommendation is different from the current one, if any are loaded
                    if (currentRecommendations.length === 0 || (currentRecommendations.length > 0 && newRecs[0].ZTFID !== currentRecommendations[currentIndex]?.ZTFID)) {
                         // And also different from the *next* one if we are not at the end of the current batch
                        if (currentIndex >= currentRecommendations.length -1 || newRecs[0].ZTFID !== currentRecommendations[currentIndex+1]?.ZTFID) {
                            // Avoid refetching if the *very next* object is what the poll returned.
                            // This means the user just hasn't swiped yet.
                            if (currentRecommendations.length > 0 && currentIndex < currentRecommendations.length -1 && newRecs[0].ZTFID === currentRecommendations[currentIndex+1]?.ZTFID){
                                //pass
                            } else {
                                showToast('New recommendations available! Refreshing...', 'info');
                                loadRecommendations(); // Reload if new data is potentially available
                            }
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Polling error:', error);
            // Optionally, stop polling if there are repeated errors
        }
    }, POLLING_INTERVAL);
    */
}

function stopPollingRecommendations() {
    if (pollingIntervalId) {
        clearInterval(pollingIntervalId);
        pollingIntervalId = null;
    }
}

// Ensure polling stops if the user navigates away or logs out
window.addEventListener('beforeunload', stopPollingRecommendations);

// If there's a logout button, add an event listener to stop polling
const logoutButton = document.querySelector('a[href="/logout"]'); // Adjust selector if needed
if (logoutButton) {
    logoutButton.addEventListener('click', stopPollingRecommendations);
}

// Function to get or generate a session ID
function getSessionId() {
    if (!currentSessionId) {
        // Generate a simple pseudo-random session ID
        currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
        console.log("Generated new session ID:", currentSessionId);
    }
    return currentSessionId;
}

async function toggleAudioRecording() {
    const recordButton = document.getElementById('recordAudioButton');
    if (!recordButton) return;

    if (isRecording) {
        // Stop recording
        mediaRecorder.stop();
        recordButton.innerHTML = '<i class="bi bi-mic-fill"></i> Record Audio';
        recordButton.classList.remove('btn-danger');
        recordButton.classList.add('btn-info');
        isRecording = false;
        showToast('Recording stopped.', 'info');
    } else {
        // Start recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = []; // Reset chunks

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // Common format, browser dependent
                const ztfid = getCurrentObject();
                if (ztfid && audioBlob.size > 0) {
                    await uploadAudioNote(ztfid, audioBlob);
                } else if (audioBlob.size === 0) {
                    showToast('No audio recorded.', 'warning');
                }                
                // Clean up media stream tracks
                stream.getTracks().forEach(track => track.stop()); 
            };

            mediaRecorder.start();
            recordButton.innerHTML = '<i class="bi bi-stop-circle-fill"></i> Stop Recording';
            recordButton.classList.remove('btn-info');
            recordButton.classList.add('btn-danger');
            isRecording = true;
            showToast('Recording started...', 'info');

        } catch (err) {
            console.error("Error accessing microphone:", err);
            showToast('Error accessing microphone. Please check permissions.', 'error');
            recordButton.innerHTML = '<i class="bi bi-mic-fill"></i> Record Audio';
            recordButton.classList.remove('btn-danger');
            recordButton.classList.add('btn-info');
            isRecording = false;
        }
    }
}

async function uploadAudioNote(ztfid, audioBlob) {
    const formData = new FormData();
    formData.append('audio_file', audioBlob, `audio_note_${ztfid}_${Date.now()}.webm`);
    const sessionId = getSessionId(); // Get or generate a session ID
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    showLoading(true);
    try {
        const csrfToken = getCookie('csrftoken');
        const response = await fetch(`/api/audio_note/${ztfid}`, {
            method: 'POST',
            headers: {
                'x-csrftoken': csrfToken
            },
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            
            // Show success popup
            showToast('Audio saved successfully!', 'success', 3000);
            
            // Load and display audio recordings for this object
            await loadAudioRecordings(ztfid);
            
        } else {
            const errorData = await response.json().catch(() => ({ detail: 'Failed to upload audio.' }));
            throw new Error(errorData.detail || response.statusText);
        }
    } catch (error) {
        console.error('Error uploading audio note:', error);
        showToast(`Error uploading audio: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

async function loadAudioRecordings(ztfid) {
    try {
        const response = await fetch(`/api/audio_notes/${ztfid}`);
        if (response.ok) {
            const recordings = await response.json();
            displayAudioRecordings(recordings);
        } else {
            console.error('Failed to load audio recordings');
        }
    } catch (error) {
        console.error('Error loading audio recordings:', error);
    }
}

function displayAudioRecordings(recordings) {
    const container = document.getElementById('audioRecordings');
    const noMessage = document.getElementById('noAudioMessage');
    const audioSection = document.getElementById('audioSection');
    
    if (!container) return;
    
    if (recordings && recordings.length > 0) {
        audioSection.style.display = 'block';
        noMessage.style.display = 'none';
        
        container.innerHTML = recordings.map(recording => createAudioElement(recording)).join('');
    } else {
        audioSection.style.display = 'block';
        noMessage.style.display = 'block';
        container.innerHTML = '';
    }
}

function createAudioElement(recording) {
    const isOwn = recording.is_own;
    const transcriptDisplay = recording.transcription ? 
        `<div class="audio-transcript mt-2">
            <small class="text-muted"><i class="bi bi-chat-text"></i> Transcript:</small>
            <p class="mb-0 small">${escapeHtml(recording.transcription)}</p>
        </div>` : '';
    
    return `
        <div class="audio-item" data-recording-id="${recording.id}">
            <div class="audio-content flex-grow-1">
                <div class="audio-header d-flex justify-content-between align-items-center mb-2">
                    <span class="audio-username ${isOwn ? 'text-primary fw-bold' : 'text-muted'}">
                        ${recording.username}${isOwn ? ' (you)' : ''}
                    </span>
                    <span class="audio-timestamp">${formatTimestamp(recording.created_at)}</span>
                </div>
                
                <div class="audio-controls">
                    <audio controls class="audio-player">
                        <source src="/api/audio_notes/file/${recording.filename}" type="audio/webm">
                        <source src="/api/audio_notes/file/${recording.filename}" type="audio/mp4">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                
                ${transcriptDisplay}
            </div>
            
            ${isOwn ? `
                <div class="audio-actions">
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteAudioRecording(${recording.id}, '${recording.ztfid}')">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            ` : ''}
        </div>
    `;
}

async function deleteAudioRecording(recordingId, ztfid) {
    if (!confirm('Are you sure you want to delete this audio recording?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/audio_notes/${recordingId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('Audio recording deleted successfully!', 'success');
            await loadAudioRecordings(ztfid);
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to delete audio recording', 'error');
        }
    } catch (error) {
        console.error('Error deleting audio recording:', error);
        showToast('Error deleting audio recording', 'error');
    }
}

// Function to populate science case tag buttons
function populateScienceCaseTagButtons() {
    const tagButtonsContainer = document.getElementById('tagButtons');
    if (tagButtonsContainer) {
        tagButtonsContainer.innerHTML = ''; // Clear any existing buttons
        SCIENCE_CASES_FOR_TAGS.forEach(scienceCase => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-outline-secondary tag-btn me-1 mb-1';
            button.dataset.tag = scienceCase;
            button.textContent = scienceCase.replace("-", " ").replace(/\b\w/g, l => l.toUpperCase()); // Format for display
            button.addEventListener('click', handleTagClick); // Existing handler
            tagButtonsContainer.appendChild(button);
        });
    }
}

async function loadExtractionStatus() {
    try {
        const response = await fetch('/api/extraction-status');
        const data = await response.json();
        
        const statusDiv = document.getElementById('extractionStatus');
        const triggerButton = document.getElementById('triggerExtraction');
        
        // Only proceed if the required elements exist
        if (!statusDiv) {
            return; // Not on a page with extraction status display
        }
        
        if (data.status === 'never_run') {
            statusDiv.innerHTML = `
                <p class="text-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    Feature extraction has never been run. Click "Update Features" to extract features from recent transients.
                </p>
            `;
            if (triggerButton) triggerButton.style.display = 'block';
        } else if (data.status === 'running') {
            const runtimeHours = data.runtime_hours || 0;
            const isStuck = data.is_stuck || false;
            
            const stuckWarning = isStuck ? `
                <div class="alert alert-warning mt-2">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Warning:</strong> Extraction has been running for ${runtimeHours.toFixed(1)} hours and may be stuck.
                    Contact an administrator if this persists.
                </div>
            ` : '';
            
            statusDiv.innerHTML = `
                <p class="text-${isStuck ? 'warning' : 'info'}">
                    <i class="fas fa-cogs fa-spin"></i>
                    Feature extraction is currently running... (${runtimeHours.toFixed(1)}h)
                </p>
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated ${isStuck ? 'bg-warning' : 'bg-info'}" 
                         role="progressbar" style="width: 70%"></div>
                </div>
                <p class="small text-muted mt-2">
                    Extraction in progress. You'll be notified when complete.
                </p>
                ${stuckWarning}
            `;
            if (triggerButton) {
                triggerButton.innerHTML = '<i class="fas fa-cogs fa-spin"></i> Running...';
                triggerButton.disabled = true;
            }
            
            // Start monitoring if not already running
            if (!extractionProgressInterval) {
                console.log('🔄 Detected running extraction, starting progress monitoring...');
                startExtractionProgressMonitoring();
            }
        } else if (data.status === 'completed') {
            const runDate = new Date(data.run_date);
            const runDateString = runDate.toLocaleString();
            const now = new Date();
            const hoursAgo = Math.max(0, (now - runDate) / (1000 * 60 * 60)).toFixed(1);
            
            statusDiv.innerHTML = `
                <p class="text-success">
                    <i class="fas fa-check-circle"></i>
                    Last extraction: ${runDateString} (${hoursAgo} hours ago)
                </p>
                <p class="text-muted">
                    Found ${data.objects_found || 0} objects, processed ${data.objects_processed || 0} in ${data.processing_time_seconds?.toFixed(1) || '0.0'}s
                </p>
            `;
            
            // Show button if extraction is old (>12 hours) OR if we're admin
            if (triggerButton) {
                if (parseFloat(hoursAgo) > 12 || window.currentUser?.is_admin) {
                    triggerButton.style.display = 'block';
                    console.log('Showing feature extraction button: hours ago =', hoursAgo, 'is_admin =', window.currentUser?.is_admin);
                } else {
                    triggerButton.style.display = 'none';
                    console.log('Hiding feature extraction button: hours ago =', hoursAgo);
                }
            }
        } else if (data.status === 'failed') {
            statusDiv.innerHTML = `
                <p class="text-danger">
                    <i class="fas fa-exclamation-circle"></i>
                    Last extraction failed: ${data.error_message || 'Unknown error'}
                </p>
            `;
            if (triggerButton) triggerButton.style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading extraction status:', error);
        const statusDiv = document.getElementById('extractionStatus');
        if (statusDiv) {
            statusDiv.innerHTML = `
                <p class="text-danger">Error loading extraction status</p>
            `;
        }
    }
}

async function triggerFeatureExtraction() {
    const triggerButton = document.getElementById('extract-features-button');
    const statusDiv = document.getElementById('extractionStatus');
    
    try {
        console.log('Starting feature extraction...');
        
        // Get user configuration
        const lookbackDaysInput = document.getElementById('lookbackDays');
        const forceReprocessInput = document.getElementById('forceReprocess');
        
        const lookbackDays = lookbackDaysInput ? parseFloat(lookbackDaysInput.value) || 7.0 : 7.0;
        const forceReprocess = forceReprocessInput ? forceReprocessInput.checked : false;
        
        console.log(`Configuration: lookback_days=${lookbackDays}, force_reprocess=${forceReprocess}`);
        
        // Validate input
        if (lookbackDays < 0.1 || lookbackDays > 365) {
            throw new Error('Lookback days must be between 0.1 and 365');
        }
        
        // Disable button and show loading state
        if (triggerButton) {
            triggerButton.disabled = true;
            triggerButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
        }
        
        // Show initial loading state
        if (statusDiv) {
            statusDiv.innerHTML = `
                <p class="text-info">
                    <i class="fas fa-spinner fa-spin"></i>
                    Starting feature extraction...
                </p>
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: 10%"></div>
                </div>
                <p class="small text-muted mt-2">
                    <i class="fas fa-info-circle"></i> Looking back ${lookbackDays} days${forceReprocess ? ' (force reprocess enabled)' : ''}
                </p>
            `;
        }
        
        // Start the extraction process
        const response = await fetch('/api/extract-features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                lookback_days: lookbackDays,
                force_reprocess: forceReprocess
            })
        });
        
        if (response.status === 403) {
            throw new Error('Feature extraction requires administrator privileges. Please contact an admin to run this operation.');
        }
        
        if (response.status === 409) {
            const errorData = await response.json();
            throw new Error(`Concurrency Error: ${errorData.detail}`);
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
                    console.log('Feature extraction started:', result);
        
        // Show progress monitoring
        if (statusDiv) {
            statusDiv.innerHTML = `
                <p class="text-info">
                    <i class="fas fa-cogs fa-spin"></i>
                    Feature extraction in progress...
                </p>
                <div class="progress mt-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" 
                         role="progressbar" style="width: 20%"></div>
                </div>
                <p class="small text-muted mt-2">
                    Started successfully! You can continue using the app while extraction runs in the background.
                </p>
            `;
        }
        
        if (triggerButton) {
            triggerButton.innerHTML = '<i class="fas fa-cogs fa-spin"></i> Running...';
        }
        
        // Show notification that extraction started
                    showToast('Feature extraction started in background! You can continue using the app.', 'success', 4000);
        
        // Start polling for completion
        startExtractionProgressMonitoring();
        
    } catch (error) {
        console.error('Error starting feature extraction:', error);
        
        if (statusDiv) {
            const isPermissionError = error.message.includes('administrator privileges');
            statusDiv.innerHTML = `
                <p class="text-${isPermissionError ? 'warning' : 'danger'}">
                    <i class="fas fa-${isPermissionError ? 'lock' : 'exclamation-circle'}"></i>
                    ${error.message}
                </p>
                ${isPermissionError ? `
                    <p class="text-muted small">
                        Only administrators can trigger feature extraction. Current admin user: <strong>agagliano</strong>
                    </p>
                ` : ''}
            `;
        }
        
        if (triggerButton) {
            triggerButton.disabled = false;
            triggerButton.innerHTML = '<i class="fas fa-cogs"></i> Extract Features';
        }
        
        showToast(error.message, error.message.includes('administrator privileges') ? 'warning' : 'error', 6000);
    }
}

// Feature extraction progress monitoring
let extractionProgressInterval = null;

function startExtractionProgressMonitoring() {
            console.log('Starting extraction progress monitoring...');
    
    // Clear any existing interval
    if (extractionProgressInterval) {
        clearInterval(extractionProgressInterval);
    }
    
    // Check status every 5 seconds
    extractionProgressInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/extraction-status');
            if (response.ok) {
                const data = await response.json();
                console.log('Extraction status:', data.status);
                
                if (data.status === 'completed') {
                    onExtractionComplete(data);
                } else if (data.status === 'failed') {
                    onExtractionFailed(data);
                } else if (data.status === 'running') {
                    updateExtractionProgress();
                }
            }
        } catch (error) {
            console.error('Error checking extraction status:', error);
        }
    }, 5000);
    
    // Also set a maximum timeout (30 minutes)
    setTimeout(() => {
        if (extractionProgressInterval) {
            console.log('⏰ Extraction monitoring timeout reached');
            clearInterval(extractionProgressInterval);
            extractionProgressInterval = null;
            showToast('⏰ Extraction monitoring timed out. Check status manually.', 'warning', 5000);
        }
    }, 30 * 60 * 1000); // 30 minutes
}

function updateExtractionProgress() {
    const statusDiv = document.getElementById('extractionStatus');
    if (statusDiv) {
        // Animate progress bar
        const progressBar = statusDiv.querySelector('.progress-bar');
        if (progressBar) {
            let currentWidth = parseInt(progressBar.style.width) || 50;
            let newWidth = Math.min(90, currentWidth + Math.random() * 10);
            progressBar.style.width = newWidth + '%';
        }
    }
}

function onExtractionComplete(data) {
    console.log('Feature extraction completed!', data);
    
    // Clear polling
    if (extractionProgressInterval) {
        clearInterval(extractionProgressInterval);
        extractionProgressInterval = null;
    }
    
    // Update UI
    const statusDiv = document.getElementById('extractionStatus');
    const triggerButton = document.getElementById('extract-features-button');
    
    if (statusDiv) {
        const objectsFound = data.objects_found || 0;
        const objectsProcessed = data.objects_processed || 0;
        const lookbackDays = data.lookback_days || 'unknown';
        
        let statusMessage;
        let statusClass;
        
        if (objectsFound === 0) {
            statusMessage = `No new objects found in the last ${lookbackDays} days`;
            statusClass = 'text-info';
        } else if (objectsProcessed === 0) {
            statusMessage = `Found ${objectsFound} objects but none needed processing (features up to date)`;
            statusClass = 'text-info';
        } else {
            statusMessage = `Feature extraction completed successfully!`;
            statusClass = 'text-success';
        }
        
        statusDiv.innerHTML = `
            <p class="${statusClass}">
                <i class="fas fa-${objectsFound === 0 ? 'info-circle' : 'check-circle'}"></i>
                ${statusMessage}
            </p>
            <div class="progress mt-2">
                <div class="progress-bar bg-${objectsFound === 0 ? 'info' : 'success'}" 
                     role="progressbar" style="width: 100%"></div>
            </div>
            <p class="text-muted mt-2">
                Found ${objectsFound} objects, processed ${objectsProcessed} 
                ${data.processing_time_seconds ? `in ${data.processing_time_seconds.toFixed(1)}s` : ''}
                <br>
                Tip: ${objectsFound === 0 ? 
                    `Try increasing lookback days or check back later for new detections.` :
                    `Features are now up to date for recent objects.`
                }
            </p>
        `;
    }
    
    if (triggerButton) {
        triggerButton.style.display = 'none';
    }
    
    // Show success notification
    showToast('Feature extraction completed successfully!', 'success', 6000);
    
    // Optionally show desktop notification if supported
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('Feature Extraction Complete', {
            body: `Processing completed! Found ${data.objects_found || 0} objects.`,
            icon: '/static/favicon.ico'
        });
    }
}

function onExtractionFailed(data) {
    console.error('Feature extraction failed:', data);
    
    // Clear polling
    if (extractionProgressInterval) {
        clearInterval(extractionProgressInterval);
        extractionProgressInterval = null;
    }
    
    // Update UI
    const statusDiv = document.getElementById('extractionStatus');
    const triggerButton = document.getElementById('extract-features-button');
    
    if (statusDiv) {
        statusDiv.innerHTML = `
            <p class="text-danger">
                <i class="fas fa-exclamation-circle"></i>
                Feature extraction failed
            </p>
            <div class="progress mt-2">
                <div class="progress-bar bg-danger" 
                     role="progressbar" style="width: 100%"></div>
            </div>
            <p class="text-muted mt-2">
                Error: ${data.error_message || 'Unknown error occurred'}
            </p>
        `;
    }
    
    if (triggerButton) {
        triggerButton.disabled = false;
        triggerButton.innerHTML = '<i class="fas fa-cogs"></i> Extract Features';
        triggerButton.style.display = 'block';
    }
    
    // Show error notification
            showToast('Feature extraction failed: ' + (data.error_message || 'Unknown error'), 'error', 8000);
}

// Request notification permissions
function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                console.log('Notification permission granted');
            }
        });
    }
}

// Cleanup function for page unload
window.addEventListener('beforeunload', () => {
    if (extractionProgressInterval) {
        clearInterval(extractionProgressInterval);
    }
});

// Comment system functions
async function loadComments(ztfid) {
    try {
        const response = await fetch(`/api/comments/${ztfid}`);
        if (response.ok) {
            const comments = await response.json();
            displayComments(comments);
        } else {
            console.error('Failed to load comments');
        }
    } catch (error) {
        console.error('Error loading comments:', error);
    }
}

function displayComments(comments) {
    const container = document.getElementById('commentsContainer');
    if (!container) return;
    
    if (comments.length === 0) {
        container.innerHTML = '<div class="no-comments">No comments yet. Be the first to comment!</div>';
        return;
    }
    
    container.innerHTML = '';
    comments.forEach(comment => {
        const commentElement = createCommentElement(comment);
        container.appendChild(commentElement);
    });
}

function createCommentElement(comment) {
    const div = document.createElement('div');
    div.className = 'comment-item';
    div.dataset.commentId = comment.id;
    
    const createdDate = new Date(comment.created_at);
    const updatedDate = new Date(comment.updated_at);
    const isEdited = createdDate.getTime() !== updatedDate.getTime();
    
    div.innerHTML = `
        <div class="comment-header">
            <span class="comment-username">${escapeHtml(comment.username)}</span>
            <span class="comment-timestamp">
                ${formatTimestamp(comment.created_at)}
                ${isEdited ? '<small>(edited)</small>' : ''}
            </span>
        </div>
        <div class="comment-text">${escapeHtml(comment.text)}</div>
        ${comment.is_own ? `
            <div class="comment-actions">
                <button class="btn btn-outline-primary btn-sm" onclick="editComment(${comment.id})">Edit</button>
                <button class="btn btn-outline-danger btn-sm" onclick="deleteComment(${comment.id})">Delete</button>
            </div>
            <div class="comment-edit-form">
                <textarea class="form-control" rows="2">${escapeHtml(comment.text)}</textarea>
                <div class="mt-2">
                    <button class="btn btn-success btn-sm" onclick="saveCommentEdit(${comment.id})">Save</button>
                    <button class="btn btn-secondary btn-sm" onclick="cancelCommentEdit(${comment.id})">Cancel</button>
                </div>
            </div>
        ` : ''}
    `;
    
    return div;
}

async function addComment() {
    const ztfid = getCurrentObject();
    if (!ztfid) return;
    
    const textarea = document.getElementById('newCommentText');
    const text = textarea.value.trim();
    
    if (!text) {
        showToast('Please enter a comment before adding.', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/comments/${ztfid}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (response.ok) {
            const newComment = await response.json();
            textarea.value = ''; // Clear the input
            
            // Add the new comment to the display
            const container = document.getElementById('commentsContainer');
            const noCommentsDiv = container.querySelector('.no-comments');
            if (noCommentsDiv) {
                container.innerHTML = '';
            }
            
            const commentElement = createCommentElement(newComment);
            container.appendChild(commentElement);
            
            showToast('Comment added successfully!', 'success');
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to add comment', 'error');
        }
    } catch (error) {
        console.error('Error adding comment:', error);
        showToast('Error adding comment', 'error');
    }
}

function editComment(commentId) {
    const commentElement = document.querySelector(`[data-comment-id="${commentId}"]`);
    if (!commentElement) return;
    
    const textDiv = commentElement.querySelector('.comment-text');
    const actionsDiv = commentElement.querySelector('.comment-actions');
    const editForm = commentElement.querySelector('.comment-edit-form');
    
    textDiv.style.display = 'none';
    actionsDiv.style.display = 'none';
    editForm.style.display = 'block';
    
    // Focus on the textarea
    const textarea = editForm.querySelector('textarea');
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function cancelCommentEdit(commentId) {
    const commentElement = document.querySelector(`[data-comment-id="${commentId}"]`);
    if (!commentElement) return;
    
    const textDiv = commentElement.querySelector('.comment-text');
    const actionsDiv = commentElement.querySelector('.comment-actions');
    const editForm = commentElement.querySelector('.comment-edit-form');
    
    textDiv.style.display = 'block';
    actionsDiv.style.display = 'flex';
    editForm.style.display = 'none';
}

async function saveCommentEdit(commentId) {
    const commentElement = document.querySelector(`[data-comment-id="${commentId}"]`);
    if (!commentElement) return;
    
    const textarea = commentElement.querySelector('.comment-edit-form textarea');
    const text = textarea.value.trim();
    
    if (!text) {
        showToast('Comment cannot be empty', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/comments/${commentId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (response.ok) {
            const updatedComment = await response.json();
            
            // Update the comment display
            const textDiv = commentElement.querySelector('.comment-text');
            const timestampSpan = commentElement.querySelector('.comment-timestamp');
            
            textDiv.textContent = updatedComment.text;
            timestampSpan.innerHTML = `${formatTimestamp(updatedComment.created_at)} <small>(edited)</small>`;
            
            cancelCommentEdit(commentId);
            showToast('Comment updated successfully!', 'success');
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to update comment', 'error');
        }
    } catch (error) {
        console.error('Error updating comment:', error);
        showToast('Error updating comment', 'error');
    }
}

async function deleteComment(commentId) {
    if (!confirm('Are you sure you want to delete this comment?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/comments/${commentId}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            // Remove the comment from the display
            const commentElement = document.querySelector(`[data-comment-id="${commentId}"]`);
            if (commentElement) {
                commentElement.remove();
                
                // Check if there are no comments left
                const container = document.getElementById('commentsContainer');
                if (container.children.length === 0) {
                    container.innerHTML = '<div class="no-comments">No comments yet. Be the first to comment!</div>';
                }
            }
            
            showToast('Comment deleted successfully!', 'success');
        } else {
            const error = await response.json();
            showToast(error.detail || 'Failed to delete comment', 'error');
        }
    } catch (error) {
        console.error('Error deleting comment:', error);
        showToast('Error deleting comment', 'error');
    }
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    
    return date.toLocaleDateString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadVoteCounts(ztfid) {
    try {
        const response = await fetch(`/api/vote-counts/${ztfid}`);
        if (response.ok) {
            const data = await response.json();
            updateVoteButtons(data.counts, data.user_vote);
            return data;
        } else {
            console.error('Failed to load vote counts');
            return null;
        }
    } catch (error) {
        console.error('Error loading vote counts:', error);
        return null;
    }
}

function updateVoteButtons(counts, userVote) {
    // Update button text with counts
    const likeBtn = document.getElementById('likeBtn');
    const dislikeBtn = document.getElementById('dislikeBtn');
    const targetBtn = document.getElementById('targetBtn');
    const skipBtn = document.getElementById('skipBtn');
    
    if (likeBtn) {
        likeBtn.innerHTML = `<i class="fas fa-thumbs-up"></i> Like (${counts.like || 0})`;
        likeBtn.classList.toggle('btn-success', userVote === 'like');
        likeBtn.classList.toggle('btn-outline-success', userVote !== 'like');
    }
    
    if (dislikeBtn) {
        dislikeBtn.innerHTML = `<i class="fas fa-thumbs-down"></i> Dislike (${counts.dislike || 0})`;
        dislikeBtn.classList.toggle('btn-danger', userVote === 'dislike');
        dislikeBtn.classList.toggle('btn-outline-danger', userVote !== 'dislike');
    }
    
    if (targetBtn) {
        targetBtn.innerHTML = `<i class="fas fa-crosshairs"></i> Target (${counts.target || 0})`;
        targetBtn.classList.toggle('btn-warning', userVote === 'target');
        targetBtn.classList.toggle('btn-outline-warning', userVote !== 'target');
    }
    
    if (skipBtn) {
        skipBtn.innerHTML = `<i class="fas fa-forward"></i> Next (${counts.skip || 0})`;
        skipBtn.classList.toggle('btn-secondary', userVote === 'skip');
        skipBtn.classList.toggle('btn-outline-secondary', userVote !== 'skip');
    }
}

function startVoteCountsRefresh() {
    // DISABLED: This automatic refresh was causing the page to disrupt user workflow
    // Users complained that the page was changing transients while they were still working
    // Vote counts will only be updated when user explicitly takes an action
    
    /*
    // Clear any existing interval
    if (voteCountsRefreshInterval) {
        clearInterval(voteCountsRefreshInterval);
    }
    
    // Refresh vote counts every 5 seconds
    voteCountsRefreshInterval = setInterval(() => {
        const obj = getCurrentObject();
        if (obj && obj.ZTFID) {
            loadVoteCounts(obj.ZTFID);
        }
    }, 5000);
    */
}

function stopVoteCountsRefresh() {
    if (voteCountsRefreshInterval) {
        clearInterval(voteCountsRefreshInterval);
        voteCountsRefreshInterval = null;
    }
}

/**
 * Set up mobile touch gestures for swipe actions
 */
function setupMobileGestures() {
    const container = document.getElementById('current-object-container');
    if (!container) return; // Only set up gestures on pages that have this element
    
    // Prevent default touch behaviors that might interfere
    container.style.touchAction = 'pan-y';
    
    container.addEventListener('touchstart', handleTouchStart, { passive: false });
    container.addEventListener('touchmove', handleTouchMove, { passive: false });
    container.addEventListener('touchend', handleTouchEnd, { passive: false });
}

/**
 * Handle touch start event
 */
function handleTouchStart(e) {
    // Only handle single finger touches
    if (e.touches.length !== 1) return;
    
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
    isSwiping = false;
    
    // Add visual feedback
    const container = e.currentTarget;
    container.style.transition = 'none';
}

/**
 * Handle touch move event
 */
function handleTouchMove(e) {
    if (e.touches.length !== 1) return;
    
    const touch = e.touches[0];
    touchEndX = touch.clientX;
    touchEndY = touch.clientY;
    
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    
    // Check if this is a swipe gesture (minimum distance)
    const minSwipeDistance = 30;
    if (Math.abs(deltaX) > minSwipeDistance || Math.abs(deltaY) > minSwipeDistance) {
        isSwiping = true;
        
        // Prevent scrolling during swipe
        e.preventDefault();
        
        // Add visual feedback during swipe
        const container = e.currentTarget;
        const maxDistance = 150;
        const progress = Math.min(1, (Math.abs(deltaX) + Math.abs(deltaY)) / maxDistance);
        
        // Determine swipe direction for visual feedback
        let swipeDirection = '';
        let backgroundColor = '';
        
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
            // Horizontal swipe
            if (deltaX > 0) {
                swipeDirection = 'like';
                backgroundColor = 'rgba(40, 167, 69, 0.2)'; // Green for like
            } else {
                swipeDirection = 'dislike';
                backgroundColor = 'rgba(220, 53, 69, 0.2)'; // Red for dislike
            }
            container.style.transform = `translateX(${deltaX * 0.3}px) rotate(${deltaX * 0.02}deg)`;
        } else {
            // Vertical swipe
            if (deltaY < 0) {
                swipeDirection = 'target';
                backgroundColor = 'rgba(255, 193, 7, 0.2)'; // Yellow for target
            } else {
                swipeDirection = 'next';
                backgroundColor = 'rgba(108, 117, 125, 0.2)'; // Gray for next
            }
            container.style.transform = `translateY(${deltaY * 0.3}px)`;
        }
        
        // Apply visual feedback
        container.style.backgroundColor = backgroundColor;
        container.style.opacity = Math.max(0.7, 1 - progress * 0.3);
        
        // Show swipe feedback icon
        showSwipeFeedback(container, swipeDirection, progress);
    }
}

/**
 * Show visual feedback during swipe
 */
function showSwipeFeedback(container, direction, progress) {
    let feedbackElement = container.querySelector('.swipe-feedback');
    if (!feedbackElement) {
        feedbackElement = document.createElement('div');
        feedbackElement.className = 'swipe-feedback';
        container.appendChild(feedbackElement);
    }
    
    const icons = {
        'like': '👍',
        'dislike': '👎',
        'target': '🎯',
        'next': '⏭️'
    };
    
    feedbackElement.textContent = icons[direction] || '';
    feedbackElement.style.opacity = Math.min(0.8, progress);
    feedbackElement.style.transform = `translate(-50%, -50%) scale(${0.5 + progress * 0.5})`;
}

/**
 * Handle touch end event
 */
function handleTouchEnd(e) {
    const container = e.currentTarget;
    
    // Reset visual feedback
    container.style.transition = 'all 0.3s ease';
    container.style.opacity = '1';
    container.style.transform = 'none';
    container.style.backgroundColor = '';
    
    // Remove swipe feedback element
    const feedbackElement = container.querySelector('.swipe-feedback');
    if (feedbackElement) {
        feedbackElement.remove();
    }
    
    if (!isSwiping) return;
    
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    
    // Determine swipe direction and minimum distance
    const minSwipeDistance = 60;
    const currentObject = getCurrentObject();
    
    if (!currentObject) return;
    
    // Check for horizontal swipes first (left/right)
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
        if (deltaX > 0) {
            // Swipe right - LIKE
                            showToast('👍 Liked!', 'success', 1500);
            handleVote(currentObject, 'like');
        } else {
            // Swipe left - DISLIKE
                            showToast('👎 Disliked!', 'error', 1500);
            handleVote(currentObject, 'dislike');
        }
    }
    // Check for vertical swipes
    else if (Math.abs(deltaY) > Math.abs(deltaX) && Math.abs(deltaY) > minSwipeDistance) {
        if (deltaY < 0) {
            // Swipe up - TARGET
                            showToast('🎯 Added to targets!', 'info', 1500);
            handleVote(currentObject, 'target');
        } else {
            // Swipe down - NEXT
            showToast('⏭️ Next!', 'info', 1500);
            handleNext(currentObject);
        }
    }
    
    // Reset touch variables
    touchStartX = 0;
    touchStartY = 0;
    touchEndX = 0;
    touchEndY = 0;
    isSwiping = false;
}

/**
 * Detect if user is on a mobile device
 */
function isMobileDevice() {
    return (typeof window.orientation !== "undefined") || (navigator.userAgent.indexOf('IEMobile') !== -1);
}

/**
 * Show mobile tutorial for first-time mobile users
 */
function showMobileTutorialIfNeeded() {
    if (!isMobileDevice()) return;
    
    const hasSeenTutorial = localStorage.getItem('mobileSwipeTutorialSeen');
    if (hasSeenTutorial) return;
    
    // Show tutorial after a short delay
    setTimeout(() => {
        showToast('📱 Mobile Tip: Swipe left/right to vote, up/down to target/skip!', 'info', 5000);
        localStorage.setItem('mobileSwipeTutorialSeen', 'true');
    }, 2000);
}

/**
 * Setup target button functionality
 */
function setupTargetModal() {
    console.log('Setting up target button...');
    
    const button = document.getElementById('targets-button');
    const modal = document.getElementById('target-modal');
    const closeBtn = document.getElementById('target-modal-close');
    
            console.log('Target elements:', {
        button: !!button,
        modal: !!modal,
        closeBtn: !!closeBtn
    });
    
    if (!button) {
        console.log('No targets button found');
        return;
    }
    
    // Set up button click handler
    button.addEventListener('click', (e) => {
                    console.log('Targets button clicked!');
        e.preventDefault();
        e.stopPropagation();
        
        if (button.disabled) {
            console.log('Button is disabled (no targets)');
            return;
        }
        
        // Check if we're on recommendations page (has modal)
        if (modal && closeBtn) {
                            console.log('Opening target modal...');
            modal.classList.add('show');
            updateTargetList();
        } else {
                            console.log('Navigating to targets page...');
            window.location.href = '/targets';
        }
    });
    
    // Set up modal functionality if it exists
    if (modal && closeBtn) {
        console.log('🎯 Setting up modal functionality');
        
        // Close modal when close button clicked
        closeBtn.addEventListener('click', (e) => {
            console.log('🎯 Close button clicked!');
            e.preventDefault();
            e.stopPropagation();
            modal.classList.remove('show');
        });
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                console.log('🎯 Clicked outside modal, closing...');
                modal.classList.remove('show');
            }
        });
        
        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('show')) {
                console.log('🎯 Escape key pressed, closing modal...');
                modal.classList.remove('show');
            }
        });
    }
    
    // Load target count on page load
    updateTargetList();
    
    console.log('✅ Target button setup complete');
}



/**
 * Clear all observing constraints
 */
function clearConstraints() {
    // Clear telescope constraint
    const telescopeSelect = document.getElementById('obsConstraintTelescope');
    if (telescopeSelect) {
        telescopeSelect.value = '';
    }
    
    // Clear days constraint
    const daysInput = document.getElementById('obsConstraintDays');
    if (daysInput) {
        daysInput.value = '';
    }
    
    // Clear magnitude limit constraint
    const magLimitInput = document.getElementById('obsConstraintMagLimit');
    if (magLimitInput) {
        magLimitInput.value = '';
    }
    
    // Clear start ZTFID constraint
    const startZtfidInput = document.getElementById('startZtfidInput');
    if (startZtfidInput) {
        startZtfidInput.value = '';
    }
    
    // Remove any existing no-objects message
    const existingMessage = document.querySelector('.no-objects-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Show toast to confirm constraints were cleared
    showToast('All constraints cleared', 'info');
    
    // Reload recommendations
    loadRecommendations();
}

/**
 * Update recommendations (e.g., after science case change)
 */
function updateRecommendations() {
    loadRecommendations();
}

// Add new functions for demo functionality
async function checkDemoAvailability() {
    try {
        console.log('🔍 Checking demo availability...');
        
        // Check if we're on the recommendations page (multiple ways to verify)
        const isRecommendationsPage = 
            window.location.pathname === '/recommendations' ||
            window.location.pathname.includes('/recommendations') ||
            document.getElementById('current-object-container') ||
            document.title.includes('Recommendations') ||
            document.querySelector('.recommendations-container');
        
        if (!isRecommendationsPage) {
            console.log('Not on recommendations page, skipping demo check');
            console.log(`   Current path: ${window.location.pathname}`);
            console.log(`   Has container: ${!!document.getElementById('current-object-container')}`);
            return;
        }
        
        console.log('✅ Confirmed on recommendations page, proceeding with demo check');
        
        // Quick check without waiting for authentication
        const response = await fetch('/api/demo/should-show');
        console.log('📡 Demo API response status:', response.status);
        
        if (!response.ok) {
            if (response.status === 401) {
                console.log('❌ User not authenticated, starting demo immediately for new users...');
                // Don't wait - start demo immediately for potential new users
                setTimeout(() => {
                    console.log('🚀 Starting immediate demo for unauthenticated user...');
                    startDemoInstant();
                }, 500);
                return;
            }
            console.error('❌ Demo API request failed:', response.status, response.statusText);
            return;
        }
        
        const data = await response.json();
        console.log('📊 Demo availability response:', data);
        
        if (data.should_show) {
            console.log('✅ Demo should show! Starting immediately...');
            console.log(`📈 User has ${data.total_votes} votes (threshold: ${data.demo_threshold})`);
            
            // Start demo immediately - no delays
            setTimeout(() => {
                console.log('🚀 Starting demo instantly...');
                startDemoInstant();
            }, 100);
        } else {
            console.log('❌ Demo should not show for this user');
            console.log(`📈 User has ${data.total_votes} votes (threshold: ${data.demo_threshold})`);
        }
    } catch (error) {
        console.error('❌ Demo check failed:', error);
        // For new users, start demo immediately as fallback
        console.log('🔄 Error occurred, starting immediate demo as fallback...');
        setTimeout(() => {
            console.log('🎯 Starting instant fallback demo...');
            startDemoInstant();
        }, 500);
    }
}

// Instant demo start with hardcoded examples (no API calls)
async function startDemoInstant() {
    try {
        console.log('🚀 Starting instant demo with hardcoded examples...');
        
        // Hardcoded demo objects that work instantly
        const instantDemoObjects = [
            {
                ztfid: 'ZTF21aaublej',
                science_case: 'snia-like',
                description: 'Type Ia supernovae are thermonuclear explosions of white dwarf stars. Notice the smooth light curve and characteristic color evolution from blue to red over ~40 days.',
                key_features: 'Fast rise (~15-20 days), characteristic color evolution, found in various galaxy types',
                enhanced_tags: {
                    science: ['snia-like', 'thermonuclear', 'standard-candle'],
                    photometry: ['fast-rise', 'blue-to-red', 'smooth-decline'],
                    host: ['various-hosts', 'spiral-arm', 'elliptical-ok']
                },
                learning_points: [
                    'Type Ia SNe typically have fast rise times (~15-20 days)',
                    'They show characteristic color evolution from blue to red',
                    'Often found in spiral galaxy arms or elliptical galaxies',
                    'Used as "standard candles" for measuring cosmic distances',
                    'Peak absolute magnitude around -19.3 in V-band'
                ],
                observing_priority: 'high',
                target_recommendation: 'Excellent target - track color evolution and get spectrum',
                coordinates: { ra: 150.1234, dec: 25.5678 },
                magnitude: 18.2,
                vote_count: 5
            },
            {
                ztfid: 'ZTF21bbcdefg',
                science_case: 'ccsn-like',
                description: 'Core-collapse supernovae result from massive stars (>8 solar masses) reaching the end of their lives. Note the plateau phase in Type IIP and evidence of hydrogen.',
                key_features: 'Plateau phase (~100 days), hydrogen signatures, star-forming regions',
                enhanced_tags: {
                    science: ['ccsn-like', 'core-collapse', 'massive-star'],
                    photometry: ['plateau-phase', 'slow-decline', 'red-colors'],
                    host: ['spiral-galaxy', 'star-forming', 'hii-regions']
                },
                learning_points: [
                    'Core-collapse SNe come from massive stars (>8 solar masses)',
                    'Type IIP shows a distinctive plateau phase lasting ~100 days',
                    'Often shows hydrogen lines in spectra (Type II)',
                    'Typically found in star-forming regions of spiral galaxies',
                    'More diverse light curve shapes than Type Ia'
                ],
                observing_priority: 'high',
                target_recommendation: 'Priority target - monitor plateau and get early spectrum',
                coordinates: { ra: 180.9876, dec: -15.4321 },
                magnitude: 17.8,
                vote_count: 3
            },
            {
                ztfid: 'ZTF21cchijkl',
                science_case: 'long-lived',
                description: 'Long-lived transients remain active for months to years, much longer than typical supernovae. These may be tidal disruption events or AGN flares.',
                key_features: 'Extended duration (months-years), nuclear location, complex light curves',
                enhanced_tags: {
                    science: ['long-lived', 'tde-candidate', 'agn-flare'],
                    photometry: ['extended-duration', 'complex-lc', 'multiple-peaks'],
                    host: ['nuclear-location', 'early-type', 'massive-bh']
                },
                learning_points: [
                    'Long-lived transients can last months to years',
                    'May be tidal disruption events (stars torn apart by black holes)',
                    'Could also be AGN variability or superluminous supernovae',
                    'Often located in galaxy centers or nuclei',
                    'Require long-term monitoring to understand'
                ],
                observing_priority: 'medium',
                target_recommendation: 'Long-term monitoring target - needs sustained follow-up',
                coordinates: { ra: 200.5432, dec: 45.1234 },
                magnitude: 19.1,
                vote_count: 2
            }
        ];
        
        console.log('✅ Instant demo objects ready, showing step-by-step demo');
        showStepByStepDemo(instantDemoObjects);
        
    } catch (error) {
        console.error('❌ Failed to start instant demo:', error);
        console.error('🔍 Error details:', error.stack);
        showToast('Failed to start instant demo - check console for details', 'error');
    }
}

// Legacy demo function (kept for potential background loading)
async function startDemo() {
    try {
        console.log('🚀 Starting demo with API content...');
        console.log('📍 Current URL:', window.location.href);
        console.log('🔐 Checking authentication status...');
        
        const response = await fetch('/api/demo/content');
        console.log('📡 Demo content API response status:', response.status);
        
        if (!response.ok) {
            console.error('❌ Demo content API request failed:', response.status, response.statusText);
            if (response.status === 401) {
                console.log('🔐 Authentication required for demo content, falling back to instant demo');
                startDemoInstant();
            } else {
                console.log('📄 Response text:', await response.text());
                showToast('Failed to load demo content, using instant demo', 'warning');
                startDemoInstant();
            }
            return;
        }
        
        const data = await response.json();
        console.log('📊 Demo content response received');
        console.log('📦 Demo objects count:', data.demo_objects ? data.demo_objects.length : 0);
        console.log('💾 Full demo data:', data);
        
        if (data.demo_objects && data.demo_objects.length > 0) {
            console.log('✅ Demo objects found, showing step-by-step demo');
            // Show step-by-step demo with real examples
            showStepByStepDemo(data.demo_objects);
        } else {
            console.error('❌ No demo objects in response, falling back to instant demo');
            console.error('📄 Response data:', data);
            startDemoInstant();
        }
    } catch (error) {
        console.error('❌ Failed to start demo:', error);
        console.error('🔍 Error details:', error.stack);
        console.log('🔄 Falling back to instant demo...');
        startDemoInstant();
    }
}

function showStepByStepDemo(demoObjects) {
    // Create larger demo modal with step-by-step interface
    const modalHtml = `
        <div class="modal fade" id="demoModal" tabindex="-1" aria-labelledby="demoModalLabel" aria-hidden="true" 
             data-bs-backdrop="static" data-bs-keyboard="false">
            <div class="modal-dialog modal-fullscreen-lg-down modal-xl-custom" style="max-width: 95vw; max-height: 95vh;">
                <div class="modal-content" style="height: 95vh;">
                    <div class="modal-header bg-success text-white">
                        <h4 class="modal-title" id="demoModalLabel">
                            <i class="bi bi-play-circle"></i> Welcome! Interactive Classification Demo
                        </h4>
                        <span class="badge bg-light text-success ms-3" id="demoStepBadge">Step 1 of ${demoObjects.length}</span>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body p-4" style="height: calc(100% - 140px); overflow-y: auto;">
                        <div class="demo-welcome-message mb-4" id="demoWelcomeMessage">
                            <div class="alert alert-info">
                                <h5><i class="fas fa-star"></i> Welcome to the Transient Classification System!</h5>
                                <p class="mb-2">You're about to learn how to classify different types of astronomical transients. 
                                We'll walk you through <strong>${demoObjects.length} real examples</strong> from our database that other users have already classified.</p>
                                <p class="mb-0"><strong>What you'll learn:</strong> How to vote, add tags, and comment on each type of transient to help advance science!</p>
                            </div>
                        </div>
                        
                        <div class="demo-step-container">
                            ${demoObjects.map((obj, index) => `
                                <div class="demo-step ${index === 0 ? 'active' : ''}" data-step="${index}">
                                    <div class="row h-100">
                                        <div class="col-lg-8">
                                            <div class="demo-science-case-header mb-3">
                                                <h3 class="text-primary">
                                                    <i class="fas fa-telescope"></i> 
                                                    ${(obj.science_case || 'transient').replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                    <span class="badge bg-primary ms-2">${obj.ztfid || obj.ZTFID || 'Demo Object'}</span>
                                                </h3>
                                                <p class="lead">${obj.description || 'This is a demonstration object to help you learn the classification system.'}</p>
                                                <div class="alert alert-light">
                                                    <strong>Key Features:</strong> ${obj.key_features || 'Practice using all the classification tools below.'}
                                                </div>
                                            </div>
                                            
                                            <div class="demo-object-viewer mb-3" style="height: 50vh; border: 3px solid #007bff; border-radius: 10px; overflow: hidden;">
                                                <iframe src="https://alerce.online/object/${obj.ztfid || obj.ZTFID || 'ZTF19aaaaaa'}" 
                                                        style="width: 100%; height: 100%; border: none;">
                                                </iframe>
                                            </div>
                                            
                                            <div class="demo-actions-section">
                                                <h5 class="mb-3"><i class="fas fa-hand-pointer"></i> Try These Actions:</h5>
                                                
                                                <!-- Enhanced Voting Section -->
                                                <div class="demo-action-group mb-3">
                                                    <h6>1. Vote on this object:</h6>
                                                    <div class="btn-group demo-vote-group mb-2" role="group">
                                                        <button class="btn btn-outline-success demo-vote-btn" data-vote="like" data-step="${index}">
                                                            <i class="fas fa-thumbs-up"></i> Like
                                                        </button>
                                                        <button class="btn btn-outline-danger demo-vote-btn" data-vote="dislike" data-step="${index}">
                                                            <i class="fas fa-thumbs-down"></i> Dislike  
                                                        </button>
                                                        <button class="btn btn-outline-warning demo-vote-btn" data-vote="target" data-step="${index}">
                                                            <i class="fas fa-crosshairs"></i> Target
                                                        </button>
                                                        <button class="btn btn-outline-secondary demo-skip-btn" data-step="${index}">
                                                            <i class="fas fa-forward"></i> Skip/Next
                                                        </button>
                                                    </div>
                                                    <div class="small text-muted">
                                                        Priority: <strong>${obj.observing_priority || 'Medium'}</strong> • 
                                                        <em>${obj.target_recommendation || 'Good practice target for learning'}</em>
                                                    </div>
                                                </div>
                                                
                                                <!-- Enhanced Tag Categories Section -->
                                                <div class="demo-action-group mb-3">
                                                    <h6>2. Add categorized tags:</h6>
                                                    
                                                    <div class="demo-tag-category mb-2">
                                                        <div class="small fw-bold text-primary mb-1">Science Classification:</div>
                                                        <div class="demo-tags-section">
                                                            ${(obj.enhanced_tags?.science || obj.tags?.science || ['snia-like', 'ccsn-like', 'long-lived']).map(tag => 
                                                                `<button class="btn btn-outline-primary btn-sm me-1 mb-1 demo-tag-btn" 
                                                                         data-tag="${tag}" data-category="science" data-step="${index}">${tag}</button>`
                                                            ).join('')}
                                                        </div>
                                                    </div>
                                                    
                                                    <div class="demo-tag-category mb-2">
                                                        <div class="small fw-bold text-warning mb-1">Photometry Features:</div>
                                                        <div class="demo-tags-section">
                                                            ${(obj.enhanced_tags?.photometry || obj.tags?.photometry || ['fast-decline', 'slowly-evolving', 'bright-peak']).map(tag => 
                                                                `<button class="btn btn-outline-warning btn-sm me-1 mb-1 demo-tag-btn" 
                                                                         data-tag="${tag}" data-category="photometry" data-step="${index}">${tag}</button>`
                                                            ).join('')}
                                                        </div>
                                                    </div>
                                                    

                                                    
                                                    <div class="demo-tag-category mb-2">
                                                        <div class="small fw-bold text-info mb-1">Host Galaxy:</div>
                                                        <div class="demo-tags-section">
                                                            ${(obj.enhanced_tags?.host || obj.tags?.host || ['early-type', 'late-type', 'nuclear']).map(tag => 
                                                                `<button class="btn btn-outline-info btn-sm me-1 mb-1 demo-tag-btn" 
                                                                         data-tag="${tag}" data-category="host" data-step="${index}">${tag}</button>`
                                                            ).join('')}
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <!-- Observing Constraints Section -->
                                                <div class="demo-action-group mb-3">
                                                    <h6>3. Set observing constraints (optional):</h6>
                                                    <div class="row g-2">
                                                        <div class="col-md-4">
                                                            <select class="form-select form-select-sm demo-telescope-select" data-step="${index}">
                                                                <option value="">Any telescope</option>
                                                                <option value="P200">Palomar 200"</option>
                                                                <option value="Keck">Keck</option>
                                                                <option value="VLT">VLT</option>
                                                            </select>
                                                        </div>
                                                        <div class="col-md-4">
                                                            <input type="number" class="form-control form-control-sm demo-mag-limit" 
                                                                   placeholder="Mag limit" min="15" max="25" step="0.1" data-step="${index}">
                                                        </div>
                                                        <div class="col-md-4">
                                                            <input type="number" class="form-control form-control-sm demo-obs-days" 
                                                                   placeholder="Days ahead" min="0" max="30" data-step="${index}">
                                                        </div>
                                                    </div>
                                                    <div class="small text-muted mt-1">Set constraints to filter future recommendations</div>
                                                </div>
                                                
                                                <!-- Comment Section -->
                                                <div class="demo-action-group mb-3">
                                                    <h6>4. Leave a comment:</h6>
                                                    <div class="input-group">
                                                        <textarea class="form-control demo-comment-text" rows="2" 
                                                                  placeholder="What do you notice about this ${obj.science_case || 'transient'}?" data-step="${index}"></textarea>
                                                        <button class="btn btn-primary demo-comment-btn" data-step="${index}">
                                                            <i class="fas fa-comment"></i> Add
                                                        </button>
                                                    </div>
                                                </div>
                                                
                                                <!-- Audio Recording Section -->
                                                <div class="demo-action-group">
                                                    <h6>5. Record audio note (optional):</h6>
                                                    <div class="d-flex align-items-center">
                                                        <button class="btn btn-outline-info demo-audio-btn" data-step="${index}">
                                                            <i class="fas fa-microphone"></i> Record Audio Note
                                                        </button>
                                                        <span class="ms-2 small text-muted">Voice notes for quick observations</span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="col-lg-4">
                                            <div class="demo-learning-panel bg-light p-3 rounded h-100">
                                                <h5 class="text-success"><i class="fas fa-graduation-cap"></i> Learning Points</h5>
                                                <ul class="demo-learning-points">
                                                    ${(obj.learning_points || [
                                                        'Practice voting on astronomical objects',
                                                        'Learn to add scientific tags and categories', 
                                                        'Try the comment system for collaboration',
                                                        'Experiment with observing constraints',
                                                        'Test audio recording functionality'
                                                    ]).map(point => 
                                                        `<li class="mb-2">${point}</li>`
                                                    ).join('')}
                                                </ul>
                                                
                                                <div class="demo-object-info mt-4">
                                                    <h6 class="text-primary">Object Details:</h6>
                                                    <div class="small">
                                                        <div><strong>ZTFID:</strong> ${obj.ztfid || obj.ZTFID || 'Demo Object'}</div>
                                                        ${(obj.coordinates || obj.ra !== undefined) ? `
                                                            <div><strong>RA:</strong> ${(obj.coordinates?.ra || obj.ra || 150.0).toFixed(4)}°</div>
                                                            <div><strong>Dec:</strong> ${(obj.coordinates?.dec || obj.dec || 25.0).toFixed(4)}°</div>
                                                        ` : `
                                                            <div><strong>RA:</strong> 150.0000°</div>
                                                            <div><strong>Dec:</strong> 25.0000°</div>
                                                        `}
                                                        ${(obj.magnitude || obj.latest_magnitude) ? `
                                                            <div><strong>Magnitude:</strong> ${(obj.magnitude || obj.latest_magnitude || 18.5).toFixed(2)}</div>
                                                        ` : `
                                                            <div><strong>Magnitude:</strong> 18.50</div>
                                                        `}
                                                        <div><strong>Votes:</strong> ${obj.vote_count || 0} users liked this example</div>
                                                    </div>
                                                </div>
                                                
                                                <div class="demo-completion-status mt-4" id="stepCompletion${index}">
                                                    <h6>Actions Completed:</h6>
                                                    <div class="demo-checklist">
                                                        <div class="demo-check-item" data-action="vote">
                                                            <i class="far fa-square"></i> Vote on object
                                                        </div>
                                                        <div class="demo-check-item" data-action="science-tag">
                                                            <i class="far fa-square"></i> Add science tag
                                                        </div>
                                                        <div class="demo-check-item" data-action="photometry-tag">
                                                            <i class="far fa-square"></i> Add photometry tag
                                                        </div>
                                                        <div class="demo-check-item" data-action="host-tag">
                                                            <i class="far fa-square"></i> Add host tag
                                                        </div>
                                                        <div class="demo-check-item" data-action="comment">
                                                            <i class="far fa-square"></i> Leave a comment
                                                        </div>
                                                        <div class="demo-check-item optional" data-action="constraints">
                                                            <i class="far fa-square"></i> Set constraints (optional)
                                                        </div>
                                                        <div class="demo-check-item optional" data-action="audio">
                                                            <i class="far fa-square"></i> Record audio (optional)
                                                        </div>
                                                    </div>
                                                    <div class="mt-2">
                                                        <div class="small text-muted">
                                                            <span class="demo-step-progress" id="stepProgress${index}">0 of 5 required actions completed</span>
                                                        </div>
                                                        <div class="progress mt-1" style="height: 6px;">
                                                            <div class="progress-bar bg-success demo-step-progress-bar" 
                                                                 id="stepProgressBar${index}" role="progressbar" 
                                                                 style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    <div class="modal-footer justify-content-between">
                        <div class="demo-progress-info">
                            <span class="text-muted">Progress: </span>
                            <span id="demoProgressText">0% complete</span>
                            <div class="progress mt-2" style="width: 200px;">
                                <div class="progress-bar bg-success" id="demoProgressBar" role="progressbar" 
                                     style="width: 0%" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                        
                        <div class="demo-navigation">
                            <button type="button" class="btn btn-secondary" id="demoPrevBtn" style="display: none;">
                                <i class="fas fa-arrow-left"></i> Previous
                            </button>
                            <button type="button" class="btn btn-primary" id="demoNextBtn">
                                Next Example <i class="fas fa-arrow-right"></i>
                            </button>
                            <button type="button" class="btn btn-success" id="demoCompleteBtn" style="display: none;">
                                <i class="fas fa-graduation-cap"></i> Complete Demo & Start Classifying!
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        .demo-step {
            display: none;
        }
        .demo-step.active {
            display: block;
        }
        .demo-check-item {
            margin-bottom: 5px;
        }
        .demo-check-item.completed i {
            color: #28a745;
        }
        .demo-check-item.completed i:before {
            content: "\\f14a"; /* fa-check-square */
        }
        .demo-check-item.optional {
            opacity: 0.7;
        }
        .demo-check-item.optional.completed {
            opacity: 1;
        }
        .demo-vote-btn.voted, .demo-tag-btn.selected, .demo-skip-btn.used {
            animation: action-success 0.6s ease-in-out;
            background-color: var(--bs-primary) !important;
            color: white !important;
            border-color: var(--bs-primary) !important;
        }
        .demo-tag-category {
            border-left: 3px solid #dee2e6;
            padding-left: 0.75rem;
            margin-bottom: 0.75rem;
        }
        .demo-tag-category:nth-child(1) { border-left-color: #0d6efd; }
        .demo-tag-category:nth-child(2) { border-left-color: #ffc107; }
        .demo-tag-category:nth-child(3) { border-left-color: #198754; }
        .demo-tag-category:nth-child(4) { border-left-color: #0dcaf0; }
        .demo-comment-text:focus, .demo-telescope-select:focus, 
        .demo-mag-limit:focus, .demo-obs-days:focus {
            border-color: #86b7fe !important;
            box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25) !important;
        }
        .demo-audio-btn.recording {
            background-color: #dc3545 !important;
            color: white !important;
            animation: pulse 1s infinite;
        }
        @keyframes action-success {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
        .modal-xl-custom {
            max-width: 95vw !important;
        }
        .demo-learning-panel {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
        }
        .demo-object-viewer {
            background: linear-gradient(45deg, #f8f9fa, #ffffff);
            border: 3px dashed #007bff !important;
        }
        </style>
    `;
    
    // Add modal to page
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // Show modal immediately (auto-open)
    const modal = new bootstrap.Modal(document.getElementById('demoModal'), {
        backdrop: 'static',  // Prevent closing by clicking outside
        keyboard: false      // Prevent closing with Esc key
    });
    modal.show();
    
    // Setup step-by-step demo interactions
    setupStepByStepDemoInteractions(demoObjects);
    
    // Remove modal when closed
    document.getElementById('demoModal').addEventListener('hidden.bs.modal', function () {
        this.remove();
    });
}

// Add classifier badge functionality
async function loadClassifierBadges(ztfid) {
    try {
        console.log(`Loading classifier badges for ${ztfid}`);
        const response = await fetch(`/api/classifier-badges/${ztfid}`);
        const data = await response.json();
        
        if (data.badges && data.badges.length > 0) {
            displayClassifierBadges(data);
            document.getElementById('classifierBadgesSection').style.display = 'block';
        } else {
            document.getElementById('classifierBadgesSection').style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading classifier badges:', error);
        document.getElementById('classifierBadgesSection').style.display = 'none';
    }
}

function displayClassifierBadges(data) {
    const container = document.getElementById('classifierBadgesData');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (!data.badges || data.badges.length === 0) {
        container.innerHTML = '<p class="text-muted">No classifier results available</p>';
        return;
    }
    
    data.badges.forEach(badge => {
        const badgeElement = document.createElement('div');
        badgeElement.className = 'classifier-badge mb-3 p-3 border rounded';
        
        // Determine badge color based on type
        let badgeClass = 'border-primary';
        let iconClass = 'fas fa-robot';
        
        if (badge.badge_type === 'anomaly') {
            badgeClass = 'border-warning';
            iconClass = 'fas fa-exclamation-triangle';
        } else if (badge.badge_type === 'classification') {
            badgeClass = 'border-success';
            iconClass = 'fas fa-check-circle';
        }
        
        badgeElement.className += ` ${badgeClass}`;
        
        badgeElement.innerHTML = `
            <div class="d-flex align-items-start">
                <i class="${iconClass} text-${badgeClass.replace('border-', '')} me-2 mt-1"></i>
                <div class="flex-grow-1">
                    <h6 class="mb-1">
                        <a href="${badge.classifier_url}" target="_blank" class="text-decoration-none">
                            ${badge.classifier_name}
                        </a>
                        ${badge.confidence ? `<span class="badge bg-${badgeClass.replace('border-', '')} ms-2">${badge.confidence.toFixed(1)}%</span>` : ''}
                    </h6>
                    <p class="mb-1 small text-muted">${badge.description}</p>
                    <p class="mb-0 small">
                        ${badge.badge_text}
                    </p>
                </div>
            </div>
        `;
        
        container.appendChild(badgeElement);
    });
}

// Add Slack voting data functionality
async function loadSlackVotes(ztfid) {
    try {
        const response = await fetch(`/api/slack-votes/${ztfid}`);
        const data = await response.json();
        
        displaySlackVotes(data);
    } catch (error) {
        console.error('Failed to load Slack votes:', error);
        // Hide the section if there's an error
        const section = document.getElementById('slackVotingSection');
        if (section) {
            section.style.display = 'none';
        }
    }
}

function displaySlackVotes(data) {
    const section = document.getElementById('slackVotingSection');
    const container = document.getElementById('slackVotingData');
    
    if (!section || !container) return;
    
    if (data.votes && data.votes.length > 0) {
        section.style.display = 'block';
        
        const votesHtml = data.votes.map(vote => `
            <div class="slack-vote-item mb-2">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <span class="fw-bold text-primary">${vote.username}</span>
                        <span class="text-muted">${vote.vote_type}</span>
                        this object in
                        <span class="badge bg-secondary">#${vote.channel}</span>
                    </div>
                    <small class="text-muted">${vote.timestamp}</small>
                </div>
            </div>
        `).join('');
        
        container.innerHTML = `
            <div class="slack-votes-list">
                ${votesHtml}
            </div>
            <div class="text-center mt-2">
                <small class="text-muted">
                    Data from ${data.source} (${data.total_count} votes)
                </small>
            </div>
        `;
    } else {
        section.style.display = 'none';
    }
}

// Add clear constraints functionality
function clearConstraints() {
    document.getElementById('obsConstraintTelescope').value = '';
    document.getElementById('obsConstraintDays').value = '';
    document.getElementById('obsConstraintMagLimit').value = '';
    
    showToast('Observing constraints cleared', 'info');
    
    // Reload recommendations with cleared constraints
    loadRecommendations();
}

function setupStepByStepDemoInteractions(demoObjects) {
    let currentStep = 0;
    
    // Enhanced progress tracking for all actions
    let demoProgress = {
        votes: new Set(),
        scienceTags: new Set(),
        photometryTags: new Set(),
        hostTags: new Set(),
        comments: new Set(),
        constraints: new Set(),
        audio: new Set(),
        skips: new Set()
    };
    
    function updateStepProgress(stepIndex = null) {
        if (stepIndex !== null) {
            // Update individual step progress
            const requiredActions = ['vote', 'science-tag', 'photometry-tag', 'host-tag', 'comment'];
            let stepCompleted = 0;
            
            const stepCompletion = document.getElementById(`stepCompletion${stepIndex}`);
            if (stepCompletion) {
                requiredActions.forEach(action => {
                    const checkItem = stepCompletion.querySelector(`[data-action="${action}"]`);
                    if (checkItem && checkItem.classList.contains('completed')) {
                        stepCompleted++;
                    }
                });
                
                const stepProgressText = document.getElementById(`stepProgress${stepIndex}`);
                const stepProgressBar = document.getElementById(`stepProgressBar${stepIndex}`);
                
                if (stepProgressText) {
                    stepProgressText.textContent = `${stepCompleted} of ${requiredActions.length} required actions completed`;
                }
                if (stepProgressBar) {
                    const stepPercentage = (stepCompleted / requiredActions.length) * 100;
                    stepProgressBar.style.width = stepPercentage + '%';
                    stepProgressBar.setAttribute('aria-valuenow', stepPercentage);
                }
            }
        }
        
        // Update overall progress - require ALL steps to be 100% complete
        let totalRequired = 0;
        let totalCompleted = 0;
        
        for (let i = 0; i < demoObjects.length; i++) {
            const requiredActions = ['vote', 'science-tag', 'photometry-tag', 'host-tag', 'comment'];
            totalRequired += requiredActions.length;
            
            const stepCompletion = document.getElementById(`stepCompletion${i}`);
            if (stepCompletion) {
                requiredActions.forEach(action => {
                    const checkItem = stepCompletion.querySelector(`[data-action="${action}"]`);
                    if (checkItem && checkItem.classList.contains('completed')) {
                        totalCompleted++;
                    }
                });
            }
        }
        
        const percentage = totalRequired > 0 ? (totalCompleted / totalRequired) * 100 : 0;
        
        const progressBar = document.getElementById('demoProgressBar');
        const progressText = document.getElementById('demoProgressText');
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.setAttribute('aria-valuenow', percentage);
        }
        if (progressText) {
            progressText.textContent = Math.round(percentage) + '% complete';
        }
        
        // Update step badge
        const stepBadge = document.getElementById('demoStepBadge');
        if (stepBadge) {
            stepBadge.textContent = `Step ${currentStep + 1} of ${demoObjects.length}`;
        }
        
        // Show completion button ONLY when 100% complete OR if user has tried everything on current step
        if (percentage >= 100) {
            document.getElementById('demoCompleteBtn').style.display = 'block';
            document.getElementById('demoNextBtn').style.display = 'none';
            
            // Update completion button text based on full completion
            const completeBtn = document.getElementById('demoCompleteBtn');
            if (completeBtn) {
                completeBtn.innerHTML = '<i class="fas fa-graduation-cap"></i> Amazing! You\'ve Mastered Everything - Start Classifying!';
                completeBtn.classList.remove('btn-success');
                completeBtn.classList.add('btn-warning');
            }
        } else if (currentStep >= demoObjects.length - 1 && percentage >= 50) {
            // If on last step and at least 50% done, allow early completion
            document.getElementById('demoCompleteBtn').style.display = 'block';
            document.getElementById('demoNextBtn').style.display = 'none';
            
            const completeBtn = document.getElementById('demoCompleteBtn');
            if (completeBtn) {
                completeBtn.innerHTML = `<i class="fas fa-graduation-cap"></i> Good Progress (${Math.round(percentage)}%) - Start Classifying!`;
            }
        }
    }
    
    function markActionCompleted(step, action) {
        const completionStatus = document.getElementById(`stepCompletion${step}`);
        if (completionStatus) {
            const checkItem = completionStatus.querySelector(`[data-action="${action}"]`);
            if (checkItem && !checkItem.classList.contains('completed')) {
                checkItem.classList.add('completed');
                
                // Show success feedback
                showToast(`✅ ${action.replace('-', ' ')} completed!`, 'success', 1500);
                
                // Update progress immediately
                updateStepProgress(step);
                updateStepProgress(); // Update overall progress
            }
        }
    }
    
    function showStep(stepIndex) {
        currentStep = stepIndex;
        
        // Hide all steps
        document.querySelectorAll('.demo-step').forEach(step => step.classList.remove('active'));
        
        // Show current step
        const currentStepElement = document.querySelector(`[data-step="${stepIndex}"]`);
        if (currentStepElement) {
            currentStepElement.classList.add('active');
        }
        
        // Update navigation buttons
        const prevBtn = document.getElementById('demoPrevBtn');
        const nextBtn = document.getElementById('demoNextBtn');
        
        if (prevBtn) {
            prevBtn.style.display = stepIndex > 0 ? 'block' : 'none';
        }
        
        if (nextBtn) {
            nextBtn.style.display = stepIndex < demoObjects.length - 1 ? 'block' : 'none';
        }
        
        // Hide welcome message after first step
        const welcomeMessage = document.getElementById('demoWelcomeMessage');
        if (welcomeMessage && stepIndex > 0) {
            welcomeMessage.style.display = 'none';
        }
        
        updateStepProgress();
    }
    
    // Enhanced event listeners for all functionality
    
    // Vote button handlers
    document.querySelectorAll('.demo-vote-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const step = parseInt(this.dataset.step);
            const voteType = this.dataset.vote;
            
            // Mark button as used
            this.classList.add('voted');
            this.innerHTML = `<i class="fas fa-check"></i> ${this.textContent.trim()}`;
            
            // Mark action completed
            markActionCompleted(step, 'vote');
            
            // Disable other vote buttons in this step
            document.querySelectorAll(`[data-step="${step}"].demo-vote-btn`).forEach(voteBtn => {
                if (voteBtn !== this) {
                    voteBtn.disabled = true;
                    voteBtn.classList.add('disabled');
                }
            });
            
            showToast(`Voted "${voteType}" - great choice!`, 'success', 2000);
        });
    });
    
    // Skip button handlers
    document.querySelectorAll('.demo-skip-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const step = parseInt(this.dataset.step);
            
            this.classList.add('used');
            this.innerHTML = '<i class="fas fa-check"></i> Skipped';
            this.disabled = true;
            
            showToast('Object skipped - sometimes you need to move on!', 'info', 2000);
        });
    });
    
    // Enhanced tag button handlers - category aware
    document.querySelectorAll('.demo-tag-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const step = parseInt(this.dataset.step);
            const tag = this.dataset.tag;
            const category = this.dataset.category;
            
            // Mark button as selected
            this.classList.add('selected');
            this.innerHTML = `<i class="fas fa-check"></i> ${tag}`;
            this.disabled = true;
            
            // Mark the corresponding category action completed
            markActionCompleted(step, `${category}-tag`);
            
            showToast(`Added ${category} tag: "${tag}"`, 'success', 1500);
        });
    });
    
    // Comment handlers
    document.querySelectorAll('.demo-comment-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const step = parseInt(this.dataset.step);
            const textarea = document.querySelector(`[data-step="${step}"].demo-comment-text`);
            
            if (textarea && textarea.value.trim()) {
                markActionCompleted(step, 'comment');
                
                // Update UI
                this.innerHTML = '<i class="fas fa-check"></i> Added';
                this.disabled = true;
                textarea.disabled = true;
                
                showToast('Comment added successfully!', 'success', 1500);
            } else {
                showToast('Please write a comment first!', 'warning', 2000);
                if (textarea) textarea.focus();
            }
        });
    });
    
    // Constraints handlers (optional)
    document.querySelectorAll('.demo-telescope-select, .demo-mag-limit, .demo-obs-days').forEach(input => {
        input.addEventListener('change', function() {
            const step = parseInt(this.dataset.step);
            
            // Check if any constraint is set for this step
            const stepConstraints = document.querySelectorAll(`[data-step="${step}"].demo-telescope-select, [data-step="${step}"].demo-mag-limit, [data-step="${step}"].demo-obs-days`);
            let hasConstraint = false;
            
            stepConstraints.forEach(constraint => {
                if (constraint.value) hasConstraint = true;
            });
            
            if (hasConstraint) {
                markActionCompleted(step, 'constraints');
                showToast('Observing constraints set!', 'info', 1500);
            }
        });
    });
    
    // Audio recording handlers (optional)
    document.querySelectorAll('.demo-audio-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const step = parseInt(this.dataset.step);
            
            if (!this.classList.contains('recording')) {
                // Start "recording"
                this.classList.add('recording');
                this.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
                
                showToast('🎤 Recording audio note...', 'info', 2000);
                
                // Auto-stop after 3 seconds (demo simulation)
                setTimeout(() => {
                    this.classList.remove('recording');
                    this.innerHTML = '<i class="fas fa-check"></i> Audio Recorded';
                    this.disabled = true;
                    
                    markActionCompleted(step, 'audio');
                    showToast('Audio note recorded!', 'success', 1500);
                }, 3000);
            }
        });
    });
    
    // Navigation handlers
    document.getElementById('demoNextBtn')?.addEventListener('click', () => {
        if (currentStep < demoObjects.length - 1) {
            showStep(currentStep + 1);
        }
    });
    
    document.getElementById('demoPrevBtn')?.addEventListener('click', () => {
        if (currentStep > 0) {
            showStep(currentStep - 1);
        }
    });
    
    // Completion handler
    document.getElementById('demoCompleteBtn')?.addEventListener('click', async () => {
        try {
            // Mark demo as completed
            await fetch('/api/demo/complete', { method: 'POST' });
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('demoModal'));
            if (modal) {
                modal.hide();
            }
            
            // Show congratulations
            showToast('🎉 Demo completed! Welcome to the classification system!', 'success', 4000);
            
            // Refresh recommendations to start real classification
            setTimeout(() => {
                loadRecommendations();
            }, 1000);
            
        } catch (error) {
            console.error('Failed to complete demo:', error);
            showToast('Demo completion recorded locally', 'info', 2000);
        }
    });
    
    // Start with first step
    showStep(0);
    
    // Add helpful hints
    setTimeout(() => {
        showToast('💡 Tip: Try all the different tag categories to learn the system!', 'info', 4000);
    }, 5000);
}

// Add manual demo trigger for testing
window.manualDemoTest = function() {
    console.log('🧪 Manual demo test triggered');
    console.log('🔍 Current page info:');
    console.log('   URL:', window.location.href);
    console.log('   Path:', window.location.pathname);
    console.log('   Title:', document.title);
    console.log('   Has current-object-container:', !!document.getElementById('current-object-container'));
    console.log('   Has recommendations-container:', !!document.querySelector('.recommendations-container'));
    
    console.log('🚀 Starting demo check...');
    checkDemoAvailability();
};

// Also add a direct demo start function
window.manualDemoStart = function() {
    console.log('🎬 Manual demo start triggered');
    startDemo();
};

// Make demo functions available globally for testing
window.checkDemoAvailability = checkDemoAvailability;
window.startDemo = startDemo;

// Function to set up lookback preset buttons
function setupLookbackPresetButtons() {
    console.log('🔧 Setting up lookback preset buttons...');
    
    const presetButtons = document.querySelectorAll('.preset-btn');
    const lookbackInput = document.getElementById('lookbackDays');
    
    if (!presetButtons.length) {
        console.warn('⚠️ No preset buttons found');
        return;
    }
    
    if (!lookbackInput) {
        console.warn('⚠️ Lookback days input not found');
        return;
    }
    
    presetButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const days = this.dataset.days;
            if (days) {
                setLookbackDays(parseFloat(days));
            }
        });
    });
    
    console.log('✅ Lookback preset buttons set up successfully');
}

// Function to set lookback days value
function setLookbackDays(days) {
    console.log(`📅 Setting lookback days to: ${days}`);
    
    const lookbackInput = document.getElementById('lookbackDays');
    if (!lookbackInput) {
        console.warn('⚠️ Lookback days input not found');
        return;
    }
    
    lookbackInput.value = days;
    
    // Visual feedback
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
        btn.classList.remove('active');
        if (parseFloat(btn.dataset.days) === days) {
            btn.classList.add('active');
            setTimeout(() => btn.classList.remove('active'), 2000);
        }
    });
    
    // Show a toast notification
    showToast(`Lookback period set to ${days} day${days !== 1 ? 's' : ''}`, 'info', 2000);
}

// Function to set up collapsible card headers
function setupCollapsibleHeaders() {
    console.log('🔧 Setting up collapsible card headers...');
    
    const collapsibleHeaders = document.querySelectorAll('.collapsible-header');
    
    if (!collapsibleHeaders.length) {
        console.warn('⚠️ No collapsible headers found');
        return;
    }
    
    collapsibleHeaders.forEach(header => {
        const targetId = header.dataset.target;
        const targetElement = document.getElementById(targetId);
        const icon = header.querySelector('.collapse-icon');
        
        if (!targetElement) {
            console.warn(`⚠️ Target element not found for ${targetId}`);
            return;
        }
        
        // Set initial aria-expanded attribute
        const isExpanded = targetElement.classList.contains('show');
        header.setAttribute('aria-expanded', isExpanded);
        
        header.addEventListener('click', function(e) {
            e.preventDefault();
            
            const isCurrentlyExpanded = header.getAttribute('aria-expanded') === 'true';
            const newExpandedState = !isCurrentlyExpanded;
            
            // Update aria-expanded attribute
            header.setAttribute('aria-expanded', newExpandedState);
            
            // Toggle the collapse
            if (newExpandedState) {
                targetElement.classList.add('show');
                console.log(`📂 Expanded: ${targetId}`);
                showToast(`Expanded ${header.querySelector('span').textContent.trim()}`, 'info', 1500);
            } else {
                targetElement.classList.remove('show');
                console.log(`📁 Collapsed: ${targetId}`);
                showToast(`Collapsed ${header.querySelector('span').textContent.trim()}`, 'info', 1500);
            }
            
            // Rotate icon
            if (icon) {
                if (newExpandedState) {
                    icon.style.transform = 'rotate(0deg)';
                } else {
                    icon.style.transform = 'rotate(-90deg)';
                }
            }
        });
    });
    
    console.log(`✅ Set up ${collapsibleHeaders.length} collapsible headers`);
}

// Function to toggle between archival and real-time recommendation modes
function toggleRealtimeMode() {
    const toggle = document.getElementById('realtimeToggle');
    const label = document.getElementById('realtimeToggleLabel');
    const description = document.getElementById('realtimeModeDescription');
    const filterRow = document.getElementById('realtimeFilterRow');
    
    if (!toggle || !label || !description || !filterRow) {
        console.warn('⚠️ Real-time toggle elements not found');
        return;
    }
    
    if (toggle.checked) {
        // Real-time mode
        label.textContent = 'Real-time';
        label.classList.add('text-warning');
        label.classList.remove('text-primary');
        description.textContent = 'Only objects with recent detections';
        filterRow.style.display = 'block';
        console.log('🔴 Switched to real-time recommendations');
        showToast('🔴 Switched to real-time recommendations', 'warning', 3000);
    } else {
        // Archival mode
        label.textContent = 'Archival';
        label.classList.add('text-primary');
        label.classList.remove('text-warning');
        description.textContent = 'Search entire archival catalog';
        filterRow.style.display = 'none';
        console.log('📚 Switched to archival recommendations');
        showToast('📚 Switched to archival recommendations', 'info', 3000);
    }
    
    // Update recommendations when mode changes
    updateRecommendations();
}