from tradingview_screener import Query,col


'''query = (Query()
 .select(
     'logoid',
     'name',
     'premarket_close',
     'premarket_change_abs',
     'premarket_change',
     'premarket_volume',
     'premarket_gap',
     'close',
     'change',
     'volume',
     'market_cap_basic',
     'Perf.1Y.MarketCap',
     'description',
     'type',
     'typespecs',
     'update_mode',
     'pricescale',
     'minmov',
     'fractional',
     'minmove2',
     'fundamental_currency_code',
     'currency',
 )
 .where(
     col('exchange').isin(['NSE']),
     col('is_primary') == True,
     col('typespecs').has('common'),
     col('typespecs').has_none_of('preferred'),
     col('type') == 'stock',
     col('premarket_change') > 3,
     col('premarket_change').not_empty(),
     col('close') >50,
     col('active_symbol') == True,
 )
 .order_by('premarket_change', ascending=False, nulls_first=False)
 .limit(100)
 .set_markets('india')
 .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
 .set_property('preset', 'pre-market-gainers'))

x = query.get_scanner_data()
for i, df in enumerate(x):
   print(df)'''
   

unusual_volume_stocks = (Query()
 .select(
     'name',
     'description',
     'logoid',
     'update_mode',
     'type',
     'typespecs',
     'relative_volume_10d_calc',
     'close',
     'pricescale',
     'minmov',
     'fractional',
     'minmove2',
     'currency',
     'change',
     'volume',
     'market_cap_basic',
     'fundamental_currency_code',
     'price_earnings_ttm',
     'earnings_per_share_diluted_ttm',
     'earnings_per_share_diluted_yoy_growth_ttm',
     'dividends_yield_current',
     'sector.tr',
     'sector',
     'market',
     'AnalystRating.tr',
     'AnalystRating',
 )
 .where(
     col('exchange').isin(['NSE']),
     col('is_primary') == True,
     col('typespecs').has('common'),
     col('typespecs').has_none_of('preferred'),
     col('type') == 'stock',
     col('active_symbol') == True,
     
 )
 .order_by('relative_volume_10d_calc', ascending=False, nulls_first=False)
 .limit(100)
 .set_markets('india')
 .set_property('symbols', {'query': {'types': ['stock', 'fund', 'dr', 'structured']}})
 .set_property('preset', 'unusual_volume')).get_scanner_data()
for i, df in enumerate(unusual_volume_stocks):
   print(df)    
