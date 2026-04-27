document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const loader = document.getElementById('loader');
    const btnText = submitBtn.querySelector('span');
    const resultsArea = document.getElementById('results-area');
    const targetScoreEl = document.getElementById('target-score');
    const recommendationsContainer = document.getElementById('recommendations-container');
    const circlePath = document.querySelector('.circle');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        btnText.style.display = 'none';
        loader.style.display = 'block';
        submitBtn.disabled = true;

        // Gather data
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = parseFloat(value) || 5.0;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                alert('Error making prediction: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to connect to the server. Make sure the Flask app is running.');
        } finally {
            // Restore button state
            btnText.style.display = 'block';
            loader.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Unhide results area
        resultsArea.classList.remove('hidden');
        resultsArea.style.display = 'grid'; // because we use grid in css for non-hidden state

        // Update Target Score
        const score = parseFloat(data.target_score);
        targetScoreEl.textContent = score.toFixed(1);

        let percentage = Math.min(100, Math.max(0, (score / 150) * 100));
        setTimeout(() => {
            circlePath.style.strokeDasharray = `${percentage}, 100`;
        }, 100);

        // Clear previous recommendations
        recommendationsContainer.innerHTML = '';

        // Add new recommendations
        data.recommendations.forEach((city, index) => {
            const item = document.createElement('div');
            item.className = 'city-item';

            item.innerHTML = `
                <div class="city-header">
                    <span class="city-name">${city.city}</span>
                    <span class="city-country">${city.country}</span>
                </div>
                <div class="city-details">
                    <span class="city-score">Score: ${city.score}</span>
                    <span>Distance: ${city.distance}</span>
                </div>
            `;

            recommendationsContainer.appendChild(item);
        });

        // Scroll to results
        setTimeout(() => {
            resultsArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }
});
