import pandas as pd

file_path = '2023-12-29.csv'
df = pd.read_csv(file_path)
df['intraday_money'] = ((df['high'] + df['low']) / 2) * df['volume']
result = df.groupby('ticker', as_index=False)['intraday_money'].sum()
result.rename(columns={'intraday_money': 'sum_intraday_money'}, inplace=True)
result = result.sort_values(by='sum_intraday_money', ascending=False)

high_liquidity_threshold_usd = 100_000_000
medium_liquidity_threshold_usd = 50_000_000

result['liquidity_class'] = result['sum_intraday_money'].apply(
    lambda x: 2 if x > high_liquidity_threshold_usd else (1 if x > medium_liquidity_threshold_usd else 0)
)

# Сортировка от большего к меньшему
result = result.sort_values(by='sum_intraday_money', ascending=False)

# Запись результата в CSV файл
output_file = 'result_with_liquidity.csv'
result.to_csv(output_file, index=False)

print(f"Результат записан в файл: {output_file}")