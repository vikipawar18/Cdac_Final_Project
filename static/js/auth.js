document.addEventListener('DOMContentLoaded', () => {
    const authForm = document.getElementById('auth-form');
    const toggleAuth = document.getElementById('toggle-auth');
    const authTitle = document.getElementById('auth-title');
    const authSubtitle = document.getElementById('auth-subtitle');
    const authSubmit = document.getElementById('auth-submit');
    const nameGroup = document.getElementById('name-group');
    const toggleText = document.getElementById('toggle-text');
    const loader = document.getElementById('loader');

    let isLogin = true;

    toggleAuth.addEventListener('click', (e) => {
        e.preventDefault();
        isLogin = !isLogin;

        if (isLogin) {
            authTitle.textContent = 'Welcome Back';
            authSubtitle.textContent = 'Please enter your details to sign in.';
            authSubmit.querySelector('span').textContent = 'Sign In';
            nameGroup.style.display = 'none';
            toggleText.innerHTML = 'Don\'t have an account? <a href="#" id="toggle-auth">Sign Up</a>';
        } else {
            authTitle.textContent = 'Create Account';
            authSubtitle.textContent = 'Join TrueSight to start analyzing videos.';
            authSubmit.querySelector('span').textContent = 'Sign Up';
            nameGroup.style.display = 'block';
            toggleText.innerHTML = 'Already have an account? <a href="#" id="toggle-auth">Sign In</a>';
        }

        // Re-attach event listener to the new toggle link
        document.getElementById('toggle-auth').addEventListener('click', arguments.callee);
    });

    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const name = document.getElementById('name').value;

        // UI Loading State
        authSubmit.disabled = true;
        authSubmit.querySelector('span').textContent = isLogin ? 'Signing In...' : 'Creating Account...';
        loader.style.display = 'inline-block';

        const endpoint = isLogin ? '/login' : '/signup';
        const payload = isLogin ? { username, password } : { username, password, name };

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (response.ok) {
                // Redirect to main page on success
                window.location.href = '/';
            } else {
                alert('Error: ' + (data.error || 'Authentication failed'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server.');
        } finally {
            // Reset Button State
            authSubmit.disabled = false;
            authSubmit.querySelector('span').textContent = isLogin ? 'Sign In' : 'Sign Up';
            loader.style.display = 'none';
        }
    });
});
