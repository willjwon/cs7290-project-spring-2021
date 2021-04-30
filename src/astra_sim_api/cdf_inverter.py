import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    result_directory = '../result/cdf'
    inverse_cdf_directory = '../result/inverse_cdf'

    if not os.path.exists(inverse_cdf_directory):
        os.makedirs(inverse_cdf_directory)

    for filename in os.listdir(path=result_directory):
        if filename.endswith('.csv'):
            cdf_file_path = os.path.join(result_directory, filename)
            cdf_data = pd.read_csv(cdf_file_path)
            del cdf_data['Unnamed: 0']

            inverse_cdf = pd.DataFrame()
            current_index = 0
            for i in range(10001):
                inverse_cdf_frequency = i / 10000
                inverse_cdf.loc[i, 'frequency'] = inverse_cdf_frequency

                current_cdf_frequency = cdf_data.loc[current_index, 'frequency']

                if inverse_cdf_frequency <= current_cdf_frequency:
                    inverse_cdf.loc[i, 'ratio'] = cdf_data.loc[current_index, 'ratio']
                else:
                    try:
                        while inverse_cdf_frequency > current_cdf_frequency:
                            current_index += 1
                            current_cdf_frequency = cdf_data.loc[current_index, 'frequency']
                        inverse_cdf.loc[i]['ratio'] = cdf_data.loc[current_index, 'ratio']
                    except KeyError:
                        inverse_cdf.loc[i, 'ratio'] = cdf_data.iloc[-1]['ratio']

            inverse_cdf_file_path = os.path.join(inverse_cdf_directory, filename.split('_')[0] + '_icdf.csv')
            inverse_cdf.to_csv(inverse_cdf_file_path)


if __name__ == '__main__':
    main()
