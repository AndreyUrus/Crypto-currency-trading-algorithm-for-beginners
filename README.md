# Cryptocurrency trading algorithm for beginners
This algorithm was written by me as an example for [**Terminus Research Hackathon**](https://github.com/TerminusResearch/hackathon). All other files to run a project and conditions can be found on the hackathon's page on [github](https://github.com/TerminusResearch/hackathon). The presented code implements the trading strategy as part of the hackathon's main project.

At the time of publication, the competition has ended and this publication does not violate any rules of the hackathon. 

**Disclaimer: The algorithm was developed only for the purpose of learning and is not a strategy that can be applied and made a profit in real trading, the author is not responsible for the results of its application. It doesn’t serve any purpose of promoting any stock or giving any specific investment advice.**

## Introduction

<div align="justify">The presented algorithm is the author's original solution. This instruction and the provided code reflect about 1/4 of the possible potential of the strategy to get into the top 10 of the hackathon leaderboard - as the minimum criterion for the success of the algorithm (minimum 175k). A strategy has been implemented that processes signals related to the “extreme” cases of the logic of price behavior proposed by the author (for more details, see the Direction of Price Movement): when there is a clear understanding of the presence of a stable price movement in the direction of growth or decline (Fig. 1), excluding making profit over shorter periods of time, which reflect the systematic behavior of the price of the traded instrument. The strategy is the result of more than 20 versions of the proposed algorithm with a total net time spent of more than 140 hours. This manual analyzes the main prerequisites, implementation features and suggestions for improving the proposed algorithm, also provides some unpromising approaches that were considered during the evolution of the algorithm.</div>

<p align="center"> <img width="695" alt="image" src="https://user-images.githubusercontent.com/114907800/224294122-a66d15d7-5595-4a7b-8756-aab465a75b0f.png">
<p align="center"> FIGURE 1. Areas showing steady growth/declinefor a relatively long time between 01/04/22-30/06/22

## Prerequisites for Logic of Algorithm
1.	Due to the fixed condition of the maximum trade size (percentage_exchange_volume = 0.03), the algorithm additionally does not consider the possible impact on the market when calculating the target_position.
2.	Due to the fixed class immutability condition «Simulator». It is impossible to additionally pass the profit value from the “simulation_data” table to the strategy algorithm, similarly to the current position (current_position), for the correct calculation of the stop-loss, therefore this metric is not in the current version of the algorithm, although it is a classic integral part of almost any trading strategy. 

    ```
    # Do not edit this code. Your submissions will anyway be executed against this exact simulator.
    …
    target_position = self.model.compute_target_position(current_data, current_position)
    ```
    
3.	In order to comply with the formal recommendation in a hackathon that a trading strategy usually consists of two components: forecasting the future price and effectively converting the price forecast into a position, several algorithm options have been implemented that give similar results, with different operating times and resource consumption: algorithm variant using LSTM-based pricing to calculate target_position and a variant with a constant approach in which there is no pricing process, and the position change is determined by a large fixed constant; the direction of trade movement in both variants of the algorithm is determined by a set of identical conditions (for more details, see the Implementation of the algorithm section). The approach is set by the ON_PRICING variable in the strategy class [default is False].
  ON_PRICING = False or True
3.1.	In the LSTM pricing approach, the current position change delta is given as ***abs(target_price - current_price)***. The price predicted by LSTM does not determine the direction of the price, since with many advantages, LSTM is not the best tool for price pricing, the main visual argument is the behavior of the predicted price: over relatively long periods of time, the price predicted by LSTM is either higher or lower than the actual price (Fig.2). However, within the framework of the proposed strategy, this shortcoming is not critical. That is why a lightweight constant approach to calculating the position change was additionally proposed, where under certain conditions the delta of the change in the current position is fixed equal to +-100.000 (for more details, see the ***Direction of Price Movement part***) - which also allows you to reduce the running time of the algorithm by an order of magnitude, avoiding expensive LSTM.
<p align="center"> <img width="526" alt="image" src="https://user-images.githubusercontent.com/114907800/224297999-d7a35101-0a5a-4ae9-b877-40536767e78a.png">
<p align="center"> FIGURE.2. Predicted Price Behavior LSTM

4.	The algorithm implements the “careful” approach to volume of position: an approach in which the algorithm minimizes the size of the position, tending it to zero at any time in the absence of signals of stable price movement (Fig. 3) (for more details, see the ***Direction of Movement Prices***) - this approach corresponds to one of the main principles of HFT: the desire to minimize the time of holding a position for maximum risk-free trading (author's opinion); the ideal situation is a completely closed position by the end of each trading day.
<p align="center"> <img width="653" alt="image" src="https://user-images.githubusercontent.com/114907800/224299529-8c0fcfac-4da3-4625-9f0f-999e98156890.png">
<p align="center"> FIGURE.3. Formation of a position only in the presence of signals

5.	The algorithm implements an approach that does not exclude relatively small losses over a sufficiently long time, which will later be compensated by "explosive" profits, as shown in Fig. 4.
<p align="center"> А (period 01/01/22 - 01/04/22) 
<p align="center"> <img width="485" alt="image" src="https://user-images.githubusercontent.com/114907800/224299703-6e2771d4-fc69-4a66-81c8-16632659dbe5.png">
<p align="center"> B (period 01/01/22 - 30/06/22)
<p align="center"> <img width="485" alt="image" src="https://user-images.githubusercontent.com/114907800/224299815-bb5a4d1c-e94f-4a86-9bb9-65ce21ccf296.png">
<p align="center"> FIGURE. 4. Profit shape on a longer period B, losses in the first half of period A insignificant relative to the final profit

6.	The current version of the algorithm has conditions with given constant thresholds for parameter values - this fact is a quick solution for the hackathon problem based on the study of metrics in the period 01/01/22 - 01/04/22 (more details in the Price movement direction part), but it is not applicable to the working version of the algorithm, where these threshold constant values must be set dynamically, otherwise, in conjunction with the absence of a stop loss, this can lead to catastrophic losses.

## Implementation of the Algorithm
Figure 5 briefly shows the block diagram of the implemented algorithm in the class **YourStrategy**
<p align="center"><img width="721" alt="image" src="https://user-images.githubusercontent.com/114907800/224300772-90bc9d7f-33c4-48d4-adf5-17fcbd5b5b2f.png">
<p align="center">FIGURE.5. Block diagram of the algorithm. Features of method A and B

The algorithm implements two approaches for calculating target_position: a fast and simple method in which there is no pricing and the position change delta is fixed and equal to ±100.000 (we will call it further - method A), and a more complex, long and resource-intensive, but containing pricing using LSTM (we will call it method B below) - in this case, the position change delta is ***± abs(target_price – current_price)***. In both options, the direction is set by the same logic, which is described in more detail below.
<p align="center"><img width="166" alt="image" src="https://user-images.githubusercontent.com/114907800/224300938-d940d1a8-ce6a-456d-ad08-3e45b0fbcfb7.png">
<p align="center">FIGURE 6. Algorithm option switch

### Method В. Peculiarities of LSTM
The variant of the algorithm containing LSTM assumes the possibility of using a ready-made model (“model_lstm_min”) or training a new one from scratch using historical data with a depth of 60,000 points. The LSTM model itself is quite classical, similar code options can be easily found on various professional forums[1], I will give just some features of the LSTM network architecture:

- define a Sequential model which consists of a linear stack of layers;
- use an open-source machine learning library, Tensorflow, to set up our LSTM network architecture;
- add a LSTM layer by giving it 100 network units. Set the return_sequence to true so that the output of the layer will be another sequence of the same length; 
- add another LSTM layer with also 100 network units. But we set the return_sequence to false for this time to only return the last output in the output sequence; 
- add a densely connected neural network layer with 25 network units;
- at last, add a densely connected layer that specifies the output of 1 network unit;
- training for 3 epochs.
<p align="center"><img width="387" alt="image" src="https://user-images.githubusercontent.com/114907800/224301221-52b91e92-7895-49a5-9e6b-f61ade3d9ff1.png">
<p align="center">FIGURE 7. Settings LSTM

### Direction of Price Movement
The algorithm introduces the concept of the current state of the price "new_part" - it is determined by the price behavior and is characterized by a delta between the 30-day and 15-day average curve. There are 4 possible price states in total (see Fig. 8):

-	«+ 10» - phase of sharp steady growth - taken with a margin of 30 units to overcome slight fluctuations;
-	«+ 15» - a phase of moderate growth in which a decline is expected in the near future and to a greater extent than in “+ 10” there is uncertainty in the price behavior;
-	«- 15» - the phase of a sharp and steady decline of price - is determined similarly to the «+10» phase with a margin of 30 units;
-	«- 10» - the phase of a moderate decline in the price, in which growth is expected in the near future and to a greater extent than in «-15» there is uncertainty in the price behavior.
<p align="center"><img width="535" alt="image" src="https://user-images.githubusercontent.com/114907800/224301756-7de14ee9-ffe1-49d0-91de-b8b91ec7cefe.png">
<p align="center">FIGURE 8. Example of price behavior 02/04/2022 and classification by phases

Further, for each phase, an individual set of features and boundary conditions is selected to maximize profits in each area, with some features:
-	all phases are united by sensitivity to changes in the current volume of the traded instrument, or rather, to the specific parameter **vol_feat** and **vol_feat_back** (Fig. 9), calculated based on data on available volumes. Boundary values and various variations of combinations of these parameters are obtained as a solution to the problem of profit maximization in each phase in the period 01/01/22 - 01/04/22;
<p align="center"><img width="756" alt="image" src="https://user-images.githubusercontent.com/114907800/224302073-a3a63e14-67c9-4fd8-a610-d90d6b6b3e9b.png">
<p align="center">FIGURE 9. Configuration of vol_feat and vol_feat_back

-	phase “-15” additionally contains a condition for limiting the signal length - this is a forced measure due to the lack of a stop loss, reflecting a conservative approach, in which the algorithm works for an extremely limited time, closing the position already during the signal (the length of the signal intervals of this phase is from 1 to 90), otherwise, due to the lack of a stop loss on trades, catastrophic losses were observed in this phase. With the right approach, this restriction must be used as an additional feature, having previously worked out the algorithm for its dynamic setting, which can be worked out both during back-testing of the strategy and at the “Paper trading[2]” stage before launching the strategy in Production;
<p align="center"><img width="633" alt="image" src="https://user-images.githubusercontent.com/114907800/224302281-978991bc-35cf-497b-a46e-8964b5ea0203.png">
<p align="center">FIGURE 10. Position delta under a non-specific condition for the current phase

-	The peculiarity of the formation of the position change delta in a sharp increase "+ 10" and "+ 15": in the absence of compliance with specific conditions (Fig. 10) contauned ***vol_feat*** and ***vol_feat_back***, the algorithm starts gereration delta change of position in the form of a sinusoid (Fig. 11) - this is the solution, which contains a slightly better result than just fixing the position at the beginning of the meeting. Due to the relatively small amount of available volume in the rapid growth phase, an position is **fixet** at the beginning of the signal, which slightly changes with a **sigmoid correction** and closes at the end of the signal (***delta_pos = - current_position***). Otherwise, with an active increase in the position, there is a great danger of realization liquidity risk, in which the position may be closed out of time and entail a large loss. This fact is a special feature of growth phases from disease phases, the logic of which is described in the next paragraph;
<p align="center"><img width="547" alt="image" src="https://user-images.githubusercontent.com/114907800/224302815-e27a3aac-e1a8-4e92-b198-00eb8762255d.png">
<p align="center">FIGURE 11. An example of a position shape in the growth phase with a sigmoid cap. Constant method A (see Fig.5, Implementation of the algorithm)

-	A feature of the formation of the delta of change of position in the decline phases "- 10" and "- 15": if there is no compliance with the specific conditions associated with ***vol_feat*** and ***vol_feat_back***, the algorithm **starts to increase** the current position, generating a delta position as a fraction of the current one (Fig. 12) - this is the solution , at which the maximum effect is achieved in the decay phase, because unlike the growth phase, when the price falls - there is no problem with the available volumes and closing the current position may not be instantaneous only because of the maximum limit on the size of the transaction (3% of the available liquidity).
<p align="center"><img width="609" alt="image" src="https://user-images.githubusercontent.com/114907800/224303326-eec3046d-c375-46c5-9396-57f7b2c85b1f.png">
<p align="center">FIGURE 12. An example of a spike-shaped shape of multidirectional positions in the decline phase: the position grows during the entire period of the signal and then closes (almost instantly). Constant method A (see Fig.5, Implementation of the algorithm)

These features are extremely important. Figure 13 shows how the profit shape of the decline phase differs with a non-specific delta position equal to 0 from the current configuration:
<p align="center"><img width="668" alt="image" src="https://user-images.githubusercontent.com/114907800/224303615-b5324a00-3a43-4c27-942a-a99f252d1184.png">
<p align="center">FIGURE 13.  Decline phase profit shape for different algorithm configurations (period 01/01/22 - 01/04/22). Constant method A (see Fig.5, Implementation of the algorithm)

Similarly, let's look at the inapplicability of the decline phase approach with a non-specific delta position in the growth phase:
<p align="center"><img width="660" alt="image" src="https://user-images.githubusercontent.com/114907800/224303836-745e1656-422e-4171-b06a-d9b487b394c0.png">
<p align="center">FIGURE 14.  Growth phase profit shape for different algorithm configurations (period 01/01/22 - 01/04/22). Constant method A (see Fig.5, Implementation of the algorithm)

Figure 15 shows the PnL shape of the final version of the algorithm in different periods:
<p align="center"><img width="586" alt="image" src="https://user-images.githubusercontent.com/114907800/224303986-169cc4b5-c477-42c8-945f-2d2ff69b574d.png">
<p align="center">FIGURE 15. Constant method A (see Fig.5, Implementation of the algorithm). Period 01/01/22 - 01/04/22, 01/04/22 - 30/06/22, 01/01/22 - 30/06/22

The algorithm works quite successfully both in the period 01/01/22-01/04/22 (profit ~40k) and 01/04/22-30/06/22 (profit ~268k), and their combinations gives an even greater effect (profit ~ 308k).


Now consider the results in the period 01/04/22-30/06/22, depending on the application of the algorithm approach (see Fig.5, ***Implementation of the algorithm***): constant method A without pricing; method B using LSTM and the finished model ('model_LSTN_min'); method B without a ready-made model with a learning process (new model):
<p align="center"><img width="656" alt="image" src="https://user-images.githubusercontent.com/114907800/224400288-4104587b-3011-4ae7-b3f8-7284e35c5ca7.png">
<p align="center">FIGURE 16. Profit shape of the proposed approaches. Period 01.04.22 – 30.06.22

Figure 16 shows the results of all three approaches: the best one is the constant version (see method A of Fig.5, ***Implementation of the algorithm***) (profit ~268k), then the pricing model (see method B of Fig.5, ***Implementation of the algorithm***) and trained by the LSTM model 'model_LSTM_new' (profit ~263k), the last place is the model with pricing without a ready model (profit ~261k) - it is important that the result of the last model may be different with a different quality of the newly trained LSTM model. ***In addition to formal compliance with the recommendation for the hackathon on the presence of pricing in algorithm, with approximately equal transaction costs, method B does not have significant advantages over method A, which is much more advantageous in terms of complexity, resource consumption and, most importantly, fast speed (see table 1)***
<p align="center"><img width="678" alt="Снимок экрана 2023-03-10 в 12 55 25" src="https://user-images.githubusercontent.com/114907800/224304808-c1d809cd-432e-4d88-852a-571f87405576.png">

## Review
Results were obtained that correspond to the conditions of the hackathon. The result of 268k corresponds to the seventh place in fixed the leaderboard (Fig. 18). Algorithm is reacting to a sharp and steady price movement. Figure 17 shows the profit shape “splash” in response to the overall price movement.

<p align="center"><img width="327" alt="image" src="https://user-images.githubusercontent.com/114907800/224312817-4c8efe1b-b6b9-4798-b80d-9fd1be27af3a.jpg">

<p align="center">FIGURE 17. PnL shape sensitivity to price movement

***Main disadvantages***: This algorithm is not optimal, but it has great potential. In this configuration of the algorithm, the boundary conditions are set constant, which is a very rough condition, but quite suitable for solving the current problem: the algorithm, in fact, is tied to the analytics of a certain period, which will certainly cause a scalability problem.

### Possible growth points: 
-	dynamic approach at boundary conditions; 
-	implementation of an approach for solving problems of balancing the worst outcome for the optimal selection of the signal period length limit = position holding time (now such a limit has been introduced in the “-15” phase - as a poor alternative to stop loss);
-	new non-marginal price phases, in which the balancing task will be even more relevant due to greater uncertainty;
-	refusal to fix the maximum trade size of the available volume by 3%, switching to more advanced approaches[3] such as TWAP, VWAP, POV, advanced models with Brownian motion and others;
-	other trading strategies with this and the instrument: trading in this instrument, under the conditions proposed within the hackathon, may be irrational due to the high volatility of the instrument (although the proposed strategy relies on the confident movement of the instrument’s price, avoiding this fact), becouse for this instrument it makes sense to focus on arbitrage strategies, strategies related to market making, less often - direction trading[4];
-	use of impulse neural networks in the algorithm.

### Rejected approaches during development: 
-	operating with classical tools such as graphical analysis, MACD approaches, etc.;
-	search for similar periods in history with the current one with the construction of further logic;
-	step-by-step transaction management: a transaction is fixed at each moment of time and is continued until a certain level of profit / loss is reached.

## Contack to me:
- Linkedin:     www.linkedin.com/in/andreyurusov/
- FB:   www.facebook.com/andrey.urusov.35
- Telegram:                                    @AndreyUrusov
- e-mail:                                 imurusov@mail.ru
    
    
## References
[1] *https://medium.com/the-handbook-of-coding-in-finance/stock-prices-prediction-using-long-short-term-memory-lstm-model-in-python-734dd1ed6827*

[2] *Irene Aldridge, «High-frequency trading: a practical guide to algorithmic strategies and trading systems» 2013, Chapter-7, «The Business of High-Frequency Trading», p.117*

[3] *Irene Aldridge, «High-frequency trading: a practical guide to algorithmic strategies and trading systems» 2013, Chapter-15, «Minimizing Market Impact», p.245*

[4] *Irene Aldridge, «High-frequency trading: a practical guide to algorithmic strategies and trading systems» 2013, Chapter-8, «Statistical Arbitrage Strategies», p.131*
