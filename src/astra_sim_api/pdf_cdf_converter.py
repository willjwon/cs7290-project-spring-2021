import pandas as pd
import os


def drop_bin(data: pd.DataFrame, threshold: float = 1) -> pd.DataFrame:
    return data[data['ratio'] <= threshold]


def pdf_to_cdf(pdf: pd.DataFrame) -> pd.DataFrame:
    cdf = pdf.copy()
    for i in range(1, len(cdf)):
        cdf.loc[i, 'frequency'] += cdf.loc[i - 1, 'frequency']
    return cdf


def main() -> None:
    result_path = '../result'
    cdf_result_path = os.path.join(result_path, 'cdf')
    if not os.path.exists(cdf_result_path):
        os.makedirs(cdf_result_path)

    for filename in os.listdir(path=result_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(result_path, filename)
            csv_data = pd.read_csv(csv_file_path)
            del csv_data['Unnamed: 0']

            csv_data = drop_bin(data=csv_data, threshold=1)
            cdf = pdf_to_cdf(pdf=csv_data)

            cdf_file_path = os.path.join(cdf_result_path, filename.split('.')[0] + '_cdf.csv')
            cdf.to_csv(cdf_file_path)


if __name__ == '__main__':
    main()