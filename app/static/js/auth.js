/* ============================================================
   AUTH UTILITIES — together platform
   ============================================================ */

var AUTH_KEY = 'together_token';

/**
 * Store JWT token after login.
 * @param {string} token
 */
function setToken(token) {
  localStorage.setItem(AUTH_KEY, token);
}

/**
 * Get the raw JWT string.
 * @returns {string|null}
 */
function getToken() {
  return localStorage.getItem(AUTH_KEY);
}

/**
 * Decode JWT payload (no verification — client-side only).
 * @returns {object|null}
 */
function getUser() {
  const token = getToken();
  if (!token) return null;
  try {
    const payload = token.split('.')[1];
    return JSON.parse(atob(payload));
  } catch {
    return null;
  }
}

/**
 * Check if token exists AND is not expired.
 * @returns {boolean}
 */
function isAuthenticated() {
  const user = getUser();
  if (!user) return false;
  if (user.exp && user.exp * 1000 < Date.now()) {
    clearToken();
    return false;
  }
  return true;
}

/**
 * Remove JWT token (sign out client-side).
 */
function clearToken() {
  localStorage.removeItem(AUTH_KEY);
}

/**
 * Guard: redirect to /login if not authenticated.
 * Call this at the top of any protected page.
 */
function requireAuth() {
  if (!isAuthenticated()) {
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    window.location.href = '/login?next=' + next;
  }
}

/**
 * Try to refresh the access token using the httpOnly refresh-token cookie.
 * Updates localStorage with the new access token on success.
 * @returns {Promise<boolean>} true if refresh succeeded
 */
async function tryRefreshToken() {
  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',
    });
    if (res.ok) {
      const data = await res.json();
      if (data.access_token) {
        setToken(data.access_token);
        return true;
      }
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Sign out: revoke refresh token on server, clear local storage, redirect.
 */
async function signOut() {
  try {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
  } catch { /* best-effort */ }
  clearToken();
  window.location.href = '/';
}

/**
 * Get display name: full_name → email prefix → 'User'
 * @returns {string}
 */
function getDisplayName() {
  const user = getUser();
  if (!user) return 'User';
  if (user.full_name) return user.full_name;
  if (user.sub) return user.sub.split('@')[0];
  return 'User';
}

/**
 * Get initials for avatar display.
 * @returns {string}
 */
function getInitials() {
  const name = getDisplayName();
  const parts = name.trim().split(' ');
  if (parts.length >= 2) return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

/**
 * Make an authenticated API request, transparently refreshing the access token
 * once on a 401 before giving up and redirecting to login.
 * @param {string} url
 * @param {object} options
 * @returns {Promise<Response>}
 */
async function authFetch(url, options = {}) {
  const token = getToken();
  let res = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      ...options.headers,
      ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    }
  });

  // On 401, attempt one silent token refresh then retry.
  if (res.status === 401 && !url.startsWith('/api/auth/')) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      const newToken = getToken();
      res = await fetch(url, {
        ...options,
        credentials: 'include',
        headers: {
          ...options.headers,
          ...(newToken ? { 'Authorization': `Bearer ${newToken}` } : {}),
        }
      });
    }

    // Still 401 after refresh attempt — session is dead.
    if (res.status === 401) {
      clearToken();
      if (!window.location.pathname.startsWith('/login')) {
        const next = encodeURIComponent(window.location.pathname + window.location.search);
        window.location.href = '/login?next=' + next;
      }
    }
  }

  return res;
}
