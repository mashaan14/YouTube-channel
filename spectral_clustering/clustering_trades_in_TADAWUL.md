# Clustering Trades in Saudi Stock Exchange (Tadawul)

Similarity graphs are superiour tools when it comes to data analysis. They become very useful when clusters are arranged in non convex shapes or containing outliers. I used a tiny dataset of 80 transactions in Saudi stock market (Tadawul) to demonestrate how similarity graphs could outperform conventinal clustering methods. Clustering stocks based on some indicators could be helpful in deversifiying a portifolio.

Let’s take a look at the data according to (price to book ratio) and (return on equity):

![data](https://github.com/user-attachments/assets/d5b398be-0745-4842-986c-1daa5d162d9e)

Here how it looks if we run k-means and told it there are 6 clusters:

![kmeans](https://github.com/user-attachments/assets/1eab62fd-5297-4be9-b7e6-f8ecb22d23a4)

Now let’s look where k-means got it wrong:

![kmeans-wrong](https://github.com/user-attachments/assets/49d0aa93-8995-4402-85b4-d3200a867cf0)

Similarity graphs required the construction of affinity matrix using a metric of choice. I used a similarity metric from reference [1], here’s the formula:

![formula](https://github.com/user-attachments/assets/2032562e-27bb-462d-b596-226b94c3d249)

Simply, it’s the distance between to points divided by multiplication of their distances to their 7th neighbour. For more information about the formula please refer to reference [1]. Obviously, the above formula will produce a fully connected graph because we’re visiting all nodes in the dataset. However, if we looked at the distribution of edges weights:

![edges](https://github.com/user-attachments/assets/1fd6f826-12bb-4369-acdb-0964db072d83)

Then exclude edges that are smaller than a specific threshold:

![edges-cut](https://github.com/user-attachments/assets/089f50e4-1bb3-428a-928f-d0cb2373bd39)

The obtained graph will be like this:

![SC](https://github.com/user-attachments/assets/efb68404-dc17-42a1-bfa8-f491f3462cb0)

And here where it goes wrong:

![SC-cut](https://github.com/user-attachments/assets/0016406a-0a75-4944-a510-15b0588f1614)

To conclude, the similarity graph helped us to detect stocks by sector and exclude outliers.

Thanks for reading.

[1] Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering." Advances in neural information processing systems. 2005.
