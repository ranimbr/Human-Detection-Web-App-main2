document.addEventListener('DOMContentLoaded', function() {
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    const cards = document.querySelectorAll('.card');
    const prevButton = document.getElementById('prev-card');
    const nextButton = document.getElementById('next-card');
    const numberOfCards = cards.length;
    let currentIndex = 0;

    // Function to update the data-index of the cards
    function updateCardIndices() {
        cards.forEach((card, index) => {
            const newIndex = (index - currentIndex + numberOfCards) % numberOfCards;
            card.dataset.index = newIndex;
        });
    }

    // Initial update of card indices
    updateCardIndices();

    // Event listener for the Next button (Corrected)
    nextButton.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + numberOfCards) % numberOfCards; // Decrement to move right
        updateCardIndices();
    });

    // Event listener for the Previous button (Corrected)
    prevButton.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % numberOfCards; // Increment to move left
        updateCardIndices();
    });

    // Theme toggle functionality (remains the same)
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            body.classList.toggle('light-theme');
            const isLightTheme = body.classList.contains('light-theme');
            localStorage.setItem('theme', isLightTheme ? 'light' : 'dark');
        });

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            body.classList.add('light-theme');
        }
    }
});