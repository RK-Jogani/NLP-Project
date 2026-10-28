const articleInput = document.querySelector('#article');
const classifyButton = document.querySelector('#classify-btn');
const sampleButton = document.querySelector('#sample-btn');
const errorMessage = document.querySelector('#error-message');
const sampleArticle = 'پاکستان کرکٹ ٹیم نے دوسرے ون ڈے میچ میں شاندار کارکردگی دکھاتے ہوئے حریف ٹیم کو شکست دے دی۔ کپتان نے اہم اننگز کھیلی اور باؤلرز نے آخری اوورز میں میچ کا رخ بدل دیا۔';

function updateCounts() {
  const text = articleInput.value;
  document.querySelector('#word-count').textContent = `${text.trim() ? text.trim().split(/\s+/).length : 0} WORDS`;
  document.querySelector('#char-count').textContent = `${text.length.toLocaleString()} / 12,000`;
}

articleInput.addEventListener('input', updateCounts);
sampleButton.addEventListener('click', () => {
  articleInput.value = sampleArticle;
  articleInput.dispatchEvent(new Event('input'));
  articleInput.focus();
});

classifyButton.addEventListener('click', async () => {
  errorMessage.textContent = '';
  const article = articleInput.value.trim();
  if (!article) {
    errorMessage.textContent = 'Please paste an Urdu news article first.';
    articleInput.focus();
    return;
  }
  classifyButton.disabled = true;
  classifyButton.querySelector('span:first-child').textContent = 'READING STORY…';
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ article }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Something went wrong. Please try again.');
    showResult(data);
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    classifyButton.disabled = false;
    classifyButton.querySelector('span:first-child').textContent = 'CLASSIFY STORY';
  }
});

function showResult(data) {
  const section = document.querySelector('#result-section');
  const primary = document.querySelector('#result-primary');
  primary.style.setProperty('--result-color', data.prediction.color);
  document.querySelector('#result-urdu').textContent = data.prediction.ur;
  document.querySelector('#result-en').textContent = data.prediction.en;
  document.querySelector('#result-time').textContent = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' }).format(new Date());
  document.querySelector('#scores').innerHTML = data.scores.map((item) => `
    <div class="score-row">
      <div class="score-label"><span>${escapeHtml(item.en)}</span><span>${item.score.toFixed(1)}%</span></div>
      <div class="score-track"><div class="score-fill" style="--bar-color:${item.color};width:${item.score}%"></div></div>
    </div>`).join('');
  const stats = data.stats;
  document.querySelector('#stats-strip').innerHTML = `<span><strong>${stats.words.toLocaleString()}</strong> WORDS</span><span><strong>${stats.characters.toLocaleString()}</strong> CHARACTERS</span><span><strong>${stats.urdu_chars.toLocaleString()}</strong> URDU SCRIPT CHARACTERS</span>`;
  section.hidden = false;
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

updateCounts();
