
import json
import matplotlib.pyplot as plt

def visualize(data_config, name, num, coin):

    file_path = (data_config.path.result_path + f'/{name}/setting{num}.json')

    # Reading the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    result = None
    
    for r in data:
        if r['asset_name'] == coin:
            result = r

    if result is None:
        print(f'no such asset tested: {coin}')
        return

    print(f"profit of coin {result['asset_name']}: {result['profit_nn']}")
    print(f"success operations of coin {result['asset_name']}: {result['successfull_operation']}")
    print(f"total operation of coin {result['asset_name']}: {result['operations']}")

    plt.plot(result['history'])
    
    plt.title('Plot of Numbers')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()
    
    prof = result['profit_nn']
    win_rate = None if result['operations'] == 0 else result['successfull_operation'] / result['operations']
    
    return prof, win_rate