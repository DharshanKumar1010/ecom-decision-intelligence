# Demo script: 5-minute walkthrough and the 12 likely questions

For the presenter. Run `streamlit run app/Home.py`. Every page has the same layout: the
question as the title, one grey sentence under it, a few key numbers, one main chart, and a
single **Technical details** expander at the bottom for anyone who wants the tables,
matrices and rules. Open that expander only when a question needs it.

**About the figures.** Every number below was checked against the repo's pipeline outputs on
2026-10-08. The dashboard computes its values live from those files, so **if the dashboard
and this script ever disagree, the dashboard wins**; re-run `python run_all.py` and update
this file.

---

## The 5-minute walkthrough

Each page: what to click, then two sentences to say. Roughly 45 seconds each.

### 0. Home (30 seconds)

*Click:* nothing; point at the page list.

> "This tool helps a marketplace decide, seller by seller, whether to keep, warn, suspend,
> feature or promote them, and it explains each decision in plain words. The clickstream and
> call data are synthetic, which is stated on the page."

*Click:* **Intelligence** in the sidebar.

### 1. Intelligence: "What is happening in the marketplace?"

*Click:* hover the funnel.

> "Of every 100 visits, about 21 add something to the cart, 13 reach checkout and 9 buy, and
> about 7.9% of delivered items arrive late. The funnel is synthetic, as the small grey tag
> says, because Olist has no real clickstream; the numbers above it are real."

### 2. Design: "How likely is an order to arrive late, and why?"

*Click:* point at the ranking score, then the bars.

> "A random forest ranks a late order above an on-time one about 78% of the time, where 50% is
> a coin flip. Its strongest signal is the month of purchase, then the seller's own past
> lateness and the distance, which shows what the model leans on, not what causes lateness."

### 3. Choice: "Which established sellers are best overall?"

*Click:* open **Technical details** and change one comparison box; the bars on the left move.

> "Sellers with at least 30 orders are scored on reviews, on-time delivery, price and volume,
> with weights from comparing the criteria in pairs, and the sentence under the chart says
> whether those judgments contradict each other. Make them contradict and it says the weights
> should not be trusted."

### 4. Discovery: "Which good sellers are we overlooking?"

*Click:* hover a hollow dot; point at the top-left **Hidden Gem** label.

> "Hidden gems are good sellers with few orders, so they are under-exposed; there are 478.
> Most of them, 434, have under 15 orders and are shown hollow, because a few good reviews
> are promising but not proof."

### 5. Implementation: "What should we do about this seller?"

*Click:* use **Try an example** to switch outcomes; open **Technical details**.

> "The page opens on a seller where a real rule fired, shows the recommendation in one word,
> and ticks off each condition behind it. Under Technical details are all the rules and what
> would have to change to get a different outcome."

### 6. DSS Architecture: "How does the system fit together?"

*Click:* point along the four boxes.

> "Data feeds models, models feed readable rules, and people see the result here and in Power
> BI. SQL Server and Power BI are the downstream layer, built by hand from the exported CSVs."

---

## The 12 questions a faculty member is most likely to ask

**1. Why a random forest, when the original plan was a simple model?**
The small models ranked late orders poorly: on the held-out test set the linear model scored
0.63 and the decision tree 0.67, against 0.78 for the forest. We only switched after a
five-fold cross-validation showed the forest's advantage was steady (0.7928, spread 0.0051).
We kept the simpler models because you can read them, and that trade-off is shown on the
Design page.

**2. Why is the clickstream and the call data synthetic?**
Olist contains orders, reviews, sellers and products, but no web events and no phone calls.
We generated both with a fixed seed so the syllabus techniques (funnel analysis, sentiment
analysis) can be demonstrated, and every place they appear carries a SYNTHETIC tag.

**3. What does an AUC of 0.78 actually mean? Is that good?**
Take one late order and one on-time order at random; the model gives the late one the higher
risk score about 78% of the time. It is clearly better than a coin flip but far from perfect.
At the default cut-off it flags about a quarter of the late orders (recall 0.25) and about
half of its flags are right (precision 0.52), so it is a ranking aid, not an oracle.

**4. Why are most hidden gems "low confidence"?**
By definition a hidden gem has high quality but low popularity, and low popularity means few
orders. 434 of the 478 hidden gems have fewer than 15 orders, just below the median of 18
orders in the scored pool. A good average on a handful of orders could be luck, so the app
marks them rather than pretending they are as certain as established sellers.

**5. Can I trust the late-risk score shown for a seller?**
Treat it as a pipeline output, not a held-out measurement. The model was trained on roughly
80% of the same order items it later scores, so a seller's average risk is likely optimistic.
That is why the app takes every model-quality claim from the held-out test set and never
turns the seller scores into claims about how many late orders we would catch.

**6. Why is "month of purchase" the strongest feature? Is that leakage?**
The month an order was placed is known when the order is made, so it does not leak the
delivery outcome, and no delivery-date column is used as a feature. It is probably picking up
particular busy or disrupted periods in this dataset, so it should not be read as a repeatable
seasonal rule.

**7. Aren't the AHP weights just opinion?**
Yes, they are judgments, which is the nature of AHP; what AHP adds is a check that the
judgments are consistent (consistency ratio 0.019, limit 0.10). The Choice page lets you
change them and watch the ranking move, and the default weights are review score 47%,
on-time rate 28%, order volume 17% and price 7%.

**8. Where do the expert-system thresholds come from?**
Each threshold is either a fixed business limit or a percentile of the real data, and the
percentile ones were recalibrated when the random forest replaced the first model, because
its risk scores spread over a wider range. The rule-by-rule justification is in
`docs/knowledge_engineering.md`, and a test fails if the thresholds drift out of line again.

**9. Does "healthier" or "good" mean anything about food or nutrition?**
No. Olist has no such data. "Good" means sound business quality: good reviews and reliable
delivery.

**10. Why keep the decision tree and the linear model if the forest is better?**
Because an ensemble cannot show a printable rule path or one readable weight per input. They
let you inspect the logic behind the numbers, at a cost in ranking quality that the Design
page states openly. The forest also has a storage cost: the saved model is about 424 MB.

**11. What are the main limits of this project?**
The clickstream and call data are synthetic. The late-risk model ranks moderately well, not
sharply. Hidden gems rest on few orders. The app is a decision aid on historical data from
one marketplace, and nothing here proves cause and effect. The speech analytics works from
transcripts only, with no audio.

**12. How do you know the live app matches the batch results, and where do SQL Server and
Power BI fit?**
The Implementation page runs the rules live, and under Technical details it compares the
result with the stored batch file for the chosen seller; a test checks 20+ sellers across all five outcomes. SQL Server and
Power BI are a separate downstream layer, built by hand from the exported CSVs, and are not
part of the Python code.
