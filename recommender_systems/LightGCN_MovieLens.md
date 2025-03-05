# Movie Recommendations using LightGCN

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/G6c6zk0RhRM" frameborder="0" allowfullscreen></iframe>
</div>

| File Name    | Description                                           |
| :--- | :---- | 
| u.data       | This is the core file. It contains ratings data: user ID, movie ID, rating, timestamp    |
| u.genre      | List of movie genres                                  |
| u.info       | Summary statistics of the dataset                     |
| u.item       | Movie information: movie ID, title, release date, genres |
| u.occupation | List of user occupations                              |
| u.user       | User information: user ID, age, gender, occupation, zip code |
| u1.base      | Training set for fold 1 of 5-fold cross-validation    |
| u1.test      | Test set for fold 1 of 5-fold cross-validation        |
| u2.base      | Training set for fold 2 of 5-fold cross-validation    |
| u2.test      | Test set for fold 2 of 5-fold cross-validation        |
| u3.base      | Training set for fold 3 of 5-fold cross-validation    |
| u3.test      | Test set for fold 3 of 5-fold cross-validation        |
| u4.base      | Training set for fold 4 of 5-fold cross-validation    |
| u4.test      | Test set for fold 4 of 5-fold cross-validation        |
| u5.base      | Training set for fold 5 of 5-fold cross-validation    |
| u5.test      | Test set for fold 5 of 5-fold cross-validation        |
| ua.base      | Additional training set split                         |
| ua.test      | Additional test set split                             |
| ub.base      | Another additional training set split                 |
| ub.test      | Another additional test set split                     |


$Recall@K = \frac{\text{Number of relevant items in top K recommendations}}{\text{Total number of relevant items}}$

<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true}, // Display math (e.g., equations on their own line)
        {left: '$', right: '$', display: false},  // Inline math (e.g., within a sentence)
        {left: '\\(', right: '\\)', display: false}, // Another way to write inline math
        {left: '\\[', right: '\\]', display: true}   // Another way to write display math
      ]
    });
  });
</script>
