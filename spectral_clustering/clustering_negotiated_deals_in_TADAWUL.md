# Clustering Negotiated Deals in Saudi Stock Exchange (TADAWUL)

The period between January 1st 2020 and May 21st 2020 was one of the most volatile periods in Saudi stock exchange (TADAWUL). There were significant drops in the index like the drops by 5.2% and 7.41% in March 16th and May 3rd respectively. In this experiment I’m trying to categorize negotiated deals that appear in this period. I’m also trying to find out did the significant drops in prices triggered high volume trades. Negotiated deals in TADAWUL are defined as share purchases negotiated between the buyer and the seller. The value of the negotiated deal does not affect the market price of the share.

The data contains 281 deals appeared between January 1st 2020 and May 21st 2020, this list can be retrieved from TADAWUL website. I also add one column to the data representing the share price in December 19th 2019. From this data I used two features:
* **First feature**: price difference between share price in May 21st 2020 and its price back in December 19th 2019. This was calculated as a percentage.
* **Second feature**: Value Traded (SAR).

![table](https://github.com/user-attachments/assets/4a8a9591-9909-4535-b5ec-61e27dc99b60)

Both features were normalized to have zero mean and unit variance. This is how data looks in a 2 dimensional space representing two features:

![Data](https://github.com/user-attachments/assets/f1dc63d4-0408-488c-83a7-c737b258d92a)

In this space, I computed pairwise similarities given this formula:

![formula](https://github.com/user-attachments/assets/b028e359-94a4-4c00-b0ce-1d6b4595a788)

The formula above is basically the distance between two points divided by multiplication of their distances to their 7th neighbor (K=7). For more information about the formula please refer to reference [1].

This formula would produce n by n pairwise similarities (i.e., 281 by 281), if we draw a histogram of all pairwise similarities it would look like this:

![BlueHist](https://github.com/user-attachments/assets/8c4a2985-95dc-4a71-b6c0-922307f70afd)

However, visualizing all these pairwise similarities as edges on a graph would be difficult. So we have to draw a threshold somewhere, I chose 0.5 to be a threshold. All pairwise similarities in the red bins would not be included in the graph:

![BlueRedHist](https://github.com/user-attachments/assets/d8808341-469c-4233-b0f2-c1dbda620f34)

If we use pairwise similarities in blue bins to construct a graph it would look like this:

![DataGraph](https://github.com/user-attachments/assets/cf607154-54d6-491d-9244-6c742594da9f)

Here is a zoomed view of the congested area:

![Zoom1](https://github.com/user-attachments/assets/73be77ca-58a2-4d74-806f-e3a7b9421407)

If we treat each subgraph as a cluster, we would have almost 15 clusters. Some clusters just contain trades on the same company at the same day. But here is a look on the most representing clusters:

![Cluster1](https://github.com/user-attachments/assets/78165476-68c9-4ed4-afd1-da782bcd91a0)

---

![Cluster2](https://github.com/user-attachments/assets/3d2ac6df-d95d-4d33-919b-e2e0c54d8b36)

---

![Cluster4](https://github.com/user-attachments/assets/ebff764e-1ea4-4d4c-86f9-2b539e90e09b)

To conclude, trades in cluster 1 have lower price difference but they appear in January and February before larger price differences occur. Cluster 2 and 4 contain trades with larger price differences and they mainly appear in March and April.

Thanks for reading.

## References
[1] Zelnik-Manor, Lihi, and Pietro Perona. "Self-tuning spectral clustering." Advances in neural information processing systems. 2005.
