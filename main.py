from D_train.i_train import conduct_experiment
from D_train.ii_backtest import conduct_backtest
from B_data.ii_db_management import sort_coin_by_size


conduct_experiment('exp_5', [5,6,7,8])

conduct_experiment('exp_7', [2,3,4,5,6,7,8])

# back test over largest 20 coins
coin_ls = sort_coin_by_size()[:20]
start = [2023, 5, 1, 10]    
end = [2023, 7, 1, 10]

conduct_backtest('exp_5', coin_ls, [5,6,7,8], start, end)

conduct_backtest('exp_7', coin_ls, [2,3,4,5,6,7,8], start, end)

