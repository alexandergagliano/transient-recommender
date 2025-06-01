/**
 * Unit tests for the frontend JavaScript
 * Run with: npm test
 */

// Mock the DOM APIs and fetch
global.fetch = jest.fn();
global.document = {
  addEventListener: jest.fn(),
  getElementById: jest.fn(),
  querySelector: jest.fn(),
  querySelectorAll: jest.fn(),
  createElement: jest.fn(),
  documentElement: {
    setAttribute: jest.fn(),
    getAttribute: jest.fn()
  },
  body: {
    appendChild: jest.fn()
  }
};
global.HTMLElement = class {};
global.localStorage = {
  getItem: jest.fn(),
  setItem: jest.fn()
};

// Import the functions to test (normally you'd use ES modules but for simplicity we'll just mock them)
let app;

// Mock functions to track calls
const mockFunctions = {
  showLoading: jest.fn(),
  loadRecommendations: jest.fn(),
  updateTargetList: jest.fn(),
  getCurrentObject: jest.fn(),
  handleVote: jest.fn(),
  handleSkip: jest.fn(),
  toggleTheme: jest.fn(),
  showToast: jest.fn(),
  removeToast: jest.fn(),
  updateObjectDisplay: jest.fn()
};

// Setup before each test
beforeEach(() => {
  // Reset all mocks
  jest.clearAllMocks();
  
  // Mock app.js functions
  app = {
    setupThemeToggle: jest.fn(),
    toggleTheme: mockFunctions.toggleTheme,
    showLoading: mockFunctions.showLoading,
    loadRecommendations: mockFunctions.loadRecommendations,
    updateTargetList: mockFunctions.updateTargetList,
    getCurrentObject: mockFunctions.getCurrentObject,
    handleVote: mockFunctions.handleVote,
    handleSkip: mockFunctions.handleSkip,
    showToast: mockFunctions.showToast,
    removeToast: mockFunctions.removeToast,
    updateObjectDisplay: mockFunctions.updateObjectDisplay
  };
  
  // Setup mock return values
  document.getElementById.mockImplementation((id) => {
    if (id === 'loadingOverlay') {
      return { style: { display: 'none' } };
    }
    if (id === 'scienceSelect') {
      return { value: 'snia-like', addEventListener: jest.fn() };
    }
    if (id === 'themeToggle') {
      return { checked: false };
    }
    if (id === 'objectNotes') {
      return { value: 'Test notes', focus: jest.fn() };
    }
    return null;
  });
  
  document.querySelector.mockImplementation((selector) => {
    if (selector === '.toast-container') {
      return { appendChild: jest.fn() };
    }
    if (selector === '.navbar-nav') {
      return { appendChild: jest.fn() };
    }
    if (selector === '.target-list') {
      return { innerHTML: '' };
    }
    return null;
  });
  
  document.querySelectorAll.mockImplementation((selector) => {
    if (selector === '.tag-btn') {
      return [{ 
        addEventListener: jest.fn(),
        dataset: { tag: 'test-tag' },
        classList: { toggle: jest.fn(), contains: jest.fn() }
      }];
    }
    if (selector === '.target-item') {
      return [{
        dataset: { 
          ztfid: 'ZTF18abcdefg',
          ra: '150.0',
          dec: '2.0'
        }
      }];
    }
    return [];
  });
  
  document.createElement.mockImplementation(() => {
    return {
      className: '',
      style: {},
      innerHTML: '',
      querySelector: jest.fn().mockReturnValue({
        addEventListener: jest.fn()
      }),
      addEventListener: jest.fn(),
      remove: jest.fn(),
      appendChild: jest.fn()
    };
  });
  
  // Mock getCurrentObject to return a test ID
  mockFunctions.getCurrentObject.mockReturnValue('ZTF18abcdefg');
  
  global.currentRecommendations = [
    { ZTFID: 'ZTF18abcdefg', ra: 150.0, dec: 2.0, latest_magnitude: 17.5 }
  ];
  global.currentIndex = 0;
  global.currentTags = new Map();
  global.currentNotes = new Map();
  
  // Setup fetch mock to resolve with different responses
  fetch.mockImplementation((url) => {
    if (url.includes('/api/recommendations')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          { ZTFID: 'ZTF18abcdefg', ra: 150.0, dec: 2.0, latest_magnitude: 17.5 }
        ])
      });
    }
    if (url.includes('/api/vote') || url.includes('/api/skip')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ status: 'success' })
      });
    }
    if (url.includes('/api/targets')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([
          { ZTFID: 'ZTF18abcdefg', ra: 150.0, dec: 2.0, latest_magnitude: 17.5 }
        ])
      });
    }
    if (url.includes('/api/tags')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(['interesting', 'bright'])
      });
    }
    if (url.includes('/api/notes')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ text: 'Test note' })
      });
    }
    return Promise.resolve({
      ok: false,
      status: 404
    });
  });
});

// Tests
describe('Theme Functionality', () => {
  test('setupThemeToggle should create theme toggle elements if they do not exist', () => {
    document.querySelector.mockReturnValueOnce(null).mockReturnValueOnce({ appendChild: jest.fn() });
    app.setupThemeToggle();
    expect(document.createElement).toHaveBeenCalled();
  });
  
  test('toggleTheme should switch between light and dark themes', () => {
    document.documentElement.getAttribute.mockReturnValue('light');
    app.toggleTheme();
    expect(document.documentElement.setAttribute).toHaveBeenCalledWith('data-theme', 'dark');
    expect(localStorage.setItem).toHaveBeenCalledWith('theme', 'dark');
    expect(mockFunctions.showToast).toHaveBeenCalledWith('Switched to dark mode', 'info');
    
    document.documentElement.getAttribute.mockReturnValue('dark');
    app.toggleTheme();
    expect(document.documentElement.setAttribute).toHaveBeenCalledWith('data-theme', 'light');
    expect(localStorage.setItem).toHaveBeenCalledWith('theme', 'light');
    expect(mockFunctions.showToast).toHaveBeenCalledWith('Switched to light mode', 'info');
  });
});

describe('Loading and Recommendations', () => {
  test('showLoading should toggle the loading overlay visibility', () => {
    app.showLoading(true);
    expect(document.getElementById('loadingOverlay').style.display).toBe('flex');
    
    app.showLoading(false);
    expect(document.getElementById('loadingOverlay').style.display).toBe('none');
  });
  
  test('loadRecommendations should fetch and update recommendations', async () => {
    await app.loadRecommendations();
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/recommendations'));
    expect(mockFunctions.showToast).toHaveBeenCalledWith('Loaded 1 recommendations', 'success');
    expect(mockFunctions.updateObjectDisplay).toHaveBeenCalled();
  });
  
  test('loadRecommendations should handle errors gracefully', async () => {
    fetch.mockImplementationOnce(() => Promise.reject(new Error('Network error')));
    await app.loadRecommendations();
    expect(mockFunctions.showToast).toHaveBeenCalledWith('Error loading recommendations. Please try again.', 'error');
  });
});

describe('User Interactions', () => {
  test('handleVote should send vote to server and update UI', async () => {
    await app.handleVote('ZTF18abcdefg', 'like');
    expect(fetch).toHaveBeenCalledWith(
      '/api/vote',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(String)
      })
    );
    expect(mockFunctions.showToast).toHaveBeenCalledWith('👍 Liked ZTF18abcdefg', 'success');
  });
  
  test('handleSkip should send skip to server and update UI', async () => {
    await app.handleSkip('ZTF18abcdefg');
    expect(fetch).toHaveBeenCalledWith(
      '/api/skip',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(String)
      })
    );
    expect(mockFunctions.showToast).toHaveBeenCalledWith('Skipped ZTF18abcdefg', 'info');
  });
});

describe('Notifications', () => {
  test('showToast should create and display a toast notification', () => {
    app.showToast('Test message', 'success');
    expect(document.createElement).toHaveBeenCalledWith('div');
    expect(document.querySelector('.toast-container').appendChild).toHaveBeenCalled();
  });
  
  test('removeToast should animate and remove a toast notification', () => {
    const mockToast = {
      style: {},
      remove: jest.fn()
    };
    jest.useFakeTimers();
    app.removeToast(mockToast);
    expect(mockToast.style.animation).toBe('slideOut 0.3s ease forwards');
    jest.runAllTimers();
    expect(mockToast.remove).toHaveBeenCalled();
  });
});

// Run the tests
if (typeof jest === 'undefined') {
  console.error('Jest is not available. Please run using npm test');
} 