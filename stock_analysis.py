from yahooquery import Screener
from datetime import datetime, timedelta, time
from scipy.interpolate import interp1d
import numpy as np
import yfinance as yf
import statistics
import pandas as pd
from collections import defaultdict
import math

MIN_30D_AVG_VOLUME = 2000000
MIN_30D_IV_HV_RATIO = 1.25
MAX_TS_SLOPE_0_45 = -0.0041

BANKROLL = 100000
KELLY_PCT = 0.15


SCREENER_NAMES = [
    'advertising_agencies',
    'aerospace_defense',
    'aggressive_small_caps',
    'agricultural_inputs',
    'airlines',
    'airports_air_services',
    'aluminum',
    'apparel_manufacturing',
    'apparel_retail',
    'asset_management',
    'auto_manufacturers',
    'auto_parts',
    'auto_truck_dealerships',
    'banks_diversified',
    'banks_regional',
    'beverages_brewers',
    'beverages_non_alcoholic',
    'beverages_wineries_distilleries',
    'biotechnology',
    'broadcasting',
    'building_materials',
    'building_products_equipment',
    'business_equipment_supplies',
    'capital_markets',
    'chemicals',
    'coking_coal',
    'communication_equipment',
    'computer_hardware',
    'confectioners',
    'conglomerates',
    'consulting_services',
    'consumer_electronics',
    'copper',
    'credit_services',
    'day_gainers',
    'day_losers',
    'department_stores',
    'diagnostics_research',
    'discount_stores',
    'drug_manufacturers_general',
    'drug_manufacturers_specialty_generic',
    'education_training_services',
    'electrical_equipment_parts',
    'electronic_components',
    'electronic_gaming_multimedia',
    'electronics_computer_distribution',
    'engineering_construction',
    'entertainment',
    'farm_heavy_construction_machinery',
    'farm_products',
    'fifty_two_wk_gainers',
    'financial_conglomerates',
    'financial_data_stock_exchanges',
    'food_distribution',
    'footwear_accessories',
    'furnishings_fixtures_appliances',
    'gambling',
    'gold',
    'grocery_stores',
    'growth_technology_stocks',
    'health_information_services',
    'healthcare_plans',
    'high_yield_high_return',
    'home_improvement_retail',
    'household_personal_products',
    'industrial_distribution',
    'information_technology_services',
    'infrastructure_operations',
    'insurance_brokers',
    'insurance_diversified',
    'insurance_life',
    'insurance_property_casualty',
    'insurance_reinsurance',
    'insurance_specialty',
    'integrated_freight_logistics',
    'internet_content_information',
    'internet_retail',
    'largest_market_cap',
    'latest_analyst_upgraded_stocks',
    'leisure',
    'lodging',
    'lumber_wood_production',
    'luxury_goods',
    'marine_shipping',
    'medical_care_facilities',
    'medical_devices',
    'medical_distribution',
    'medical_instruments_supplies',
    'mega_cap_hc',
    'metal_fabrication',
    'morningstar_five_star_stocks',
    'mortgage_finance',
    'most_actives',
    'most_institutionally_bought_large_cap_stocks',
    'most_institutionally_held_large_cap_stocks',
    'most_institutionally_sold_large_cap_stocks',
    'most_shorted_stocks',
    'most_visited',
    'most_visited_basic_materials',
    'most_visited_communication_services',
    'most_visited_consumer_cyclical',
    'most_visited_consumer_defensive',
    'most_visited_energy',
    'most_visited_financial_services',
    'most_visited_healthcare',
    'most_visited_industrials',
    'most_visited_real_estate',
    'most_visited_technology',
    'most_visited_utilities',
    'most_watched_tickers',
    'ms_basic_materials',
    'ms_communication_services',
    'ms_consumer_cyclical',
    'ms_consumer_defensive',
    'ms_energy',
    'ms_financial_services',
    'ms_healthcare',
    'ms_industrials',
    'ms_real_estate',
    'ms_technology',
    'ms_utilities',
    'net_net_strategy',
    'oil_gas_drilling',
    'oil_gas_e_p',
    'oil_gas_equipment_services',
    'oil_gas_integrated',
    'oil_gas_midstream',
    'oil_gas_refining_marketing',
    'other_industrial_metals_mining',
    'other_precious_metals_mining',
    'packaged_foods',
    'packaging_containers',
    'paper_paper_products',
    'personal_services',
    'pharmaceutical_retailers',
    'pollution_treatment_controls',
    'portfolio_actions_most_added',
    'portfolio_actions_most_deleted',
    'portfolio_anchors',
    'publishing',
    'railroads',
    'real_estate_development',
    'real_estate_diversified',
    'real_estate_services',
    'recreational_vehicles',
    'reit_diversified',
    'reit_healthcare_facilities',
    'reit_hotel_motel',
    'reit_industrial',
    'reit_mortgage',
    'reit_office',
    'reit_residential',
    'reit_retail',
    'reit_specialty',
    'rental_leasing_services',
    'residential_construction',
    'resorts_casinos',
    'restaurants',
    'scientific_technical_instruments',
    'security_protection_services',
    'semiconductor_equipment_materials',
    'semiconductors',
    'shell_companies',
    'silver',
    'small_cap_gainers',
    'software_application',
    'software_infrastructure',
    'solar',
    'specialty_business_services',
    'specialty_chemicals',
    'specialty_industrial_machinery',
    'specialty_retail',
    'staffing_employment_services',
    'steel',
    'stocks_most_bought_by_hedge_funds',
    'stocks_most_bought_by_pension_fund',
    'stocks_most_bought_by_private_equity',
    'stocks_most_bought_by_sovereign_wealth_fund',
    'stocks_with_most_institutional_buyers',
    'stocks_with_most_institutional_sellers',
    'strong_undervalued_stocks',
    'telecom_services',
    'textile_manufacturing',
    'thermal_coal',
    'tobacco',
    'tools_accessories',
    'top_energy_us',
    'top_options_implied_volatality',
    'top_options_open_interest',
    'top_stocks_owned_by_cathie_wood',
    'top_stocks_owned_by_goldman_sachs',
    'top_stocks_owned_by_ray_dalio',
    'top_stocks_owned_by_warren_buffet',
    'travel_services',
    'trucking',
    'undervalued_growth_stocks',
    'undervalued_large_caps',
    'undervalued_wide_moat_stocks',
    'uranium',
    'utilities_diversified',
    'utilities_independent_power_producers',
    'utilities_regulated_electric',
    'utilities_regulated_gas',
    'utilities_regulated_water',
    'utilities_renewable',
    'waste_management',
]


def sort_and_filter_to_next_45d(dates, today: datetime):
    return sorted([datetime.strptime(date, "%Y-%m-%d").date() for date in dates if datetime.strptime(date, "%Y-%m-%d").date() <= today.date() + timedelta(days=45) and datetime.strptime(date, "%Y-%m-%d").date() != today.date()])


# Calculates historical volatility using the Yang-Zhang equations: see https://www.jstor.org/stable/10.1086/209650?seq=6
def calculate_30d_historical_volatility(price_data):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=30,
        center=False
    ).sum() * (1.0 / (30 - 1.0))

    open_vol = log_oc_sq.rolling(
        window=30,
        center=False
    ).sum() * (1.0 / (30 - 1.0))

    window_rs = rs.rolling(
        window=30,
        center=False
    ).sum() * (1.0 / (30 - 1.0))

    k = 0.34 / (1.34 + ((30 + 1) / (30 - 1)))
    result = (open_vol + k * close_vol + (1 - k) *
              window_rs).apply(np.sqrt) * np.sqrt(252)

    return result.iloc[-1]


def naive_calculate_historical_volatility(price_data):
    daily_returns = []
    for i in range(1, len(price_data)):
        daily_returns.append(
            (price_data.iloc[i]['Close']-price_data.iloc[i-1]['Close'])/price_data.iloc[i-1]['Close'])
    stdd = statistics.stdev(daily_returns[-30:])
    ann_vol = stdd * np.sqrt(252)
    return ann_vol


def gemini_calculate_yang_zhang_volatility(df, period="30d"):
    end_date = df.index[-1]  # Get the last date in the DataFrame
    start_date = end_date - pd.Timedelta(period)  # Calculate the start date

    # Create a boolean mask to select the desired period
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_period = df.loc[mask]

    if df_period.empty:
        print(f"No data found for the specified period: {period}")
        return None

    # Calculate overnight volatility
    overnight_volatility = np.log(
        df_period['Open'] / df_period['Close'].shift(1))

    # Calculate Rogers-Satchell volatility
    rs_volatility = (np.log(df_period['High'] / df_period['Open']) * (np.log(df_period['High'] / df_period['Close'])) +
                     np.log(df_period['Low'] / df_period['Open']) * (np.log(df_period['Low'] / df_period['Close'])))

    # Calculate open-to-close volatility
    open_to_close_volatility = np.log(df_period['Close'] / df_period['Open'])

    # Calculate Yang-Zhang volatility (using typical weights)
    k = 0.34  # You can adjust this weight
    yang_zhang_volatility = np.sqrt((overnight_volatility.var() + k * open_to_close_volatility.var() +
                                     # Annualize
                                     (1 - k) * rs_volatility.mean()) * 252)

    return yang_zhang_volatility


def build_term_structure(days, impl_vols):
    days = np.array(days)
    impl_vols = np.array(impl_vols)

    sort_idx = days.argsort()
    days = days[sort_idx]
    impl_vols = impl_vols[sort_idx]

    spline = interp1d(days, impl_vols, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return impl_vols[0]
        elif dte > days[-1]:
            return impl_vols[-1]
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(ticker):
    todays_data = ticker.history(period='1d')
    return todays_data['Close'].iloc[0]


print("finding stocks with earnings calls in the next week... ", end="")
screener = Screener()

upcoming_earnings_by_enter_datetime = defaultdict(list)
today = datetime.today()
one_week_later = today + timedelta(days=7)

tickers_with_price_and_timestamps = {}
already_added_tickers = []

for screener_name in SCREENER_NAMES:
    data = screener.get_screeners(screener_name, count=250)
    for stock in data.get(screener_name, {}).get("quotes", []):
        if "earningsCallTimestampStart" in stock:
            earnings_datetime = datetime.fromtimestamp(
                stock["earningsCallTimestampStart"])
            # we will target earnings calls that:
            #  - are in the next 7 days
            #  - we know the exact time and day of (are not estimates), and
            #  - occur from Monday AMC through Friday BMO
            #
            # We disregard earnings calls that occur BMO on Monday and AMC on Friday, since that would mean we would hold our position for longer than 18h
            is_earnings_on_monday_after_market_hours = earnings_datetime.weekday(
            ) == 0 and earnings_datetime.time() > time(15, 59)
            is_earnings_on_tue_thru_outside_of_market_hours = earnings_datetime.weekday() > 0 and earnings_datetime.weekday(
            ) < 4 and (earnings_datetime.time() > time(15, 59) or earnings_datetime.time() < time(7, 1))
            is_earnings_on_friday_before_market_hours = earnings_datetime.weekday(
            ) == 4 and earnings_datetime.time() < time(7, 1)
            if today <= earnings_datetime <= one_week_later and (is_earnings_on_monday_after_market_hours or is_earnings_on_tue_thru_outside_of_market_hours or is_earnings_on_friday_before_market_hours):
                ticker = stock["symbol"]
                is_earnings_date_an_estimate = stock["isEarningsDateEstimate"]
                if ticker not in already_added_tickers and not is_earnings_date_an_estimate:
                    enter_position_time = datetime(1970, 1, 1)
                    exit_position_time = datetime(1970, 1, 1)
                    if earnings_datetime.time() > time(15, 59):
                        enter_position_time = earnings_datetime.replace(
                            hour=14, minute=45)
                        exit_position_time = earnings_datetime.replace(
                            hour=8, minute=45) + timedelta(days=1)
                    elif earnings_datetime.time() < time(7, 1):
                        enter_position_time = earnings_datetime.replace(
                            hour=14, minute=45) - timedelta(days=1)
                        exit_position_time = earnings_datetime.replace(
                            hour=8, minute=45)
                    tickers_with_price_and_timestamps[ticker] = {
                        "price": stock["regularMarketPrice"],
                        "earnings_call_datetime": earnings_datetime,
                        "enter_position_time": enter_position_time,
                        "exit_position_time": exit_position_time,
                    }
                    already_added_tickers.append(ticker)
print("done!")
# retrieve data using yfinance
stocks = yf.Tickers(list(tickers_with_price_and_timestamps.keys()))
for symbol in stocks.tickers:
    print(f"considering {symbol}... ", end="")
    single_stock = stocks.tickers[symbol]
    # get the expiration dates for the options listed that are:
    #  - not today, and
    #  - not excessively in the future
    filtered_exp_dates = sort_and_filter_to_next_45d(
        list(single_stock.options), today)
    # make sure there are at least 4 provided expiration dates for the stock in the net 50d. this is a rough proxy for "does the stock have weekly options available"
    # we do this because we want to be as close as possible to targeting a 30d spread on our calendar, and without weekly options we are unlikely to be able to do that
    # note that I also think "are weekly options available" is a rough proxy for "does this stock have alot of trading volume", which is a critera we check later anyway
    if len(filtered_exp_dates) < 4:
        print("skipped! (no weekly options)")
        continue
    # get the option chains for each of the expiration dates
    options_chains = {exp_date: single_stock.option_chain(
        exp_date.strftime('%Y-%m-%d')) for exp_date in filtered_exp_dates}
    underlying_price = get_current_price(
        single_stock) if tickers_with_price_and_timestamps[symbol]["price"] is None else tickers_with_price_and_timestamps[symbol]["price"]
    if underlying_price is None:
        print("skipped! (cannot determine stock price)")
        continue
    at_the_money_impl_vols = {}
    strike_price = 0.0
    front_sell_call_date = filtered_exp_dates[0]
    front_sell_call_price = 0.0
    back_buy_call_date = filtered_exp_dates[0] + timedelta(days=28)
    back_buy_call_price = 0.0
    for exp_date, chain in options_chains.items():
        calls = chain.calls
        puts = chain.puts

        if calls.empty or puts.empty:
            continue

        # For this chain, figure out call entry and which put entry is closest to the stock's actual price (a.k.a. at-the-money), then grab the implied volatility for those entries and average them together.
        # Generally speaking, IV is lowest at-the-money (https://en.wikipedia.org/wiki/Volatility_smile)
        call_abs_diffs = (calls['strike'] - underlying_price).abs()
        call_min_diff_idx = call_abs_diffs.idxmin()
        call_impl_vol = calls.loc[call_min_diff_idx, 'impliedVolatility']
        put_abs_diffs = (puts['strike'] - underlying_price).abs()
        put_min_diff_idx = put_abs_diffs.idxmin()
        put_impl_vol = puts.loc[put_min_diff_idx, 'impliedVolatility']
        at_the_money_impl_vols[exp_date] = (call_impl_vol + put_impl_vol) / 2.0
        strike_price = calls.loc[call_min_diff_idx, 'strike']

        # Save the prices of the call if it is the same date as the target front or back call
        if exp_date == front_sell_call_date:
            front_sell_call_price = (calls.loc[call_min_diff_idx, 'bid'] + calls.loc[call_min_diff_idx, 'ask']) / 2
        elif exp_date == back_buy_call_date:
            back_buy_call_price = (calls.loc[call_min_diff_idx, 'bid'] + calls.loc[call_min_diff_idx, 'ask']) / 2

    overall_price_per_unit = back_buy_call_price - front_sell_call_price

    if not at_the_money_impl_vols:
        # TODO: should we have a certain number of at_the_money_impl_vols instead of just checking whether it is empty?
        print("skipped! (no values for at-the-money implied volatility! (need to figure out when this occurs and how we can prevent it))")
        continue
    days_to_expiry_entries = []
    impl_vols = []
    for exp_date, impl_vol in at_the_money_impl_vols.items():
        days_to_expiry = (exp_date - today.date()).days
        days_to_expiry_entries.append(days_to_expiry)
        impl_vols.append(impl_vol)

    calculate_implied_volatility = build_term_structure(
        days_to_expiry_entries, impl_vols)

    # Calculate the slope of the line that crosses through the volatility value 45 days out, as well as the volatility value at the soonest expiry date
    ts_slope_0_45 = (calculate_implied_volatility(
        45) - calculate_implied_volatility(days_to_expiry_entries[0])) / (45-days_to_expiry_entries[0])

    price_history = single_stock.history(period='3mo')
    iv30_hv30 = calculate_implied_volatility(
        30) / calculate_30d_historical_volatility(price_history)

    avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

    # We do not want to consider stocks that have low average trading volume
    if (avg_volume < MIN_30D_AVG_VOLUME):
        print("skipped! (not enough average volume)")
        continue
    # We do not want to consider stocks that have an implied volatility that is too low compared to their historical volatility over the last 30 days
    if (iv30_hv30 < MIN_30D_IV_HV_RATIO):
        print("skipped! (iv/hv too low)")
        continue
    # We do not want to consider stocks that have a next-45-day term structure slope that is too high
    if (ts_slope_0_45 > MAX_TS_SLOPE_0_45):
        print("skipped! (ts_slope_0_45 too high)")
        continue
    # create an somewhat-arbitrary score that we assign to the trade so that we can rank them later on
    score = (avg_volume / 300000000.0) + (iv30_hv30 / 35.0) + (ts_slope_0_45 / -0.17)
    enter_position_time = datetime(1970, 1, 1)
    exit_position_time = datetime(1970, 1, 1)
    if earnings_datetime.time() > time(15, 59):
        enter_position_time = earnings_datetime.replace(hour=14, minute=45)
        exit_position_time = earnings_datetime.replace(
            hour=8, minute=45) + timedelta(days=1)
    elif earnings_datetime.time() < time(7, 1):
        enter_position_time = earnings_datetime.replace(
            hour=14, minute=45) - timedelta(days=1)
        exit_position_time = earnings_datetime.replace(hour=8, minute=45)
    upcoming_earnings_by_enter_datetime[tickers_with_price_and_timestamps[symbol]["enter_position_time"]].append({
        "symbol": symbol,
        "underlying_price": round(underlying_price, 2),
        "strike_price": strike_price,
        "earnings_call_datetime": tickers_with_price_and_timestamps[symbol]["earnings_call_datetime"],
        "enter_position_time": tickers_with_price_and_timestamps[symbol]["enter_position_time"],
        "exit_position_time": tickers_with_price_and_timestamps[symbol]["exit_position_time"],
        "price_per_unit": overall_price_per_unit,
        "front_sell_call_date": front_sell_call_date,
        "back_buy_call_date": back_buy_call_date,
        "score": score
    })
    print("added!")

for key in sorted(upcoming_earnings_by_enter_datetime):
    print(f"[{key.strftime('%m/%d')}]")
    remaining_bankroll = BANKROLL
    for upcoming_earning in sorted(upcoming_earnings_by_enter_datetime[key], key=lambda x: x["score"], reverse=True):
        overall_price_per_option = math.ceil(upcoming_earning["price_per_unit"] * 100) + (1.3 * 2) # note: options are made up of 100 units a piece, and brokers charge $1.30 for both entering and existing a position. we also assume that the brokers round UP to the nearest cent for each unit, which puts us on the conservative side 
        how_many_options_to_buy = -999999 if overall_price_per_option < 0.01 else math.floor((remaining_bankroll * KELLY_PCT) / overall_price_per_option)
        print(f"({upcoming_earning['score']:.4f}) - buy {how_many_options_to_buy} units of {upcoming_earning['symbol']} @ ${upcoming_earning['strike_price']:.2f} (underlying: ${upcoming_earning['underlying_price']:.2f}) -- [calendar] {upcoming_earning['front_sell_call_date'].strftime('%m/%d')} (sell) {upcoming_earning['back_buy_call_date'].strftime('%m/%d')} (buy) -- open @ {upcoming_earning['enter_position_time'].strftime('%H:%M on %m/%d')}, close @ {upcoming_earning['exit_position_time'].strftime('%H:%M on %m/%d')} (earnings @ {upcoming_earning['earnings_call_datetime'].strftime('%H:%M on %m/%d')})")
        remaining_bankroll -= math.ceil(float(how_many_options_to_buy) * overall_price_per_option) - 1.3 * how_many_options_to_buy

# TODO: if you backtest, make sure you don't get overrun by the fees
