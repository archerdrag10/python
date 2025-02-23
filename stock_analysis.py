from yahooquery import Screener
from datetime import datetime, timedelta, time
from scipy.interpolate import interp1d
import numpy as np
import yfinance as yf

MIN_30D_AVG_VOLUME = 2000000
MIN_30D_IV_RV = 1.25
MAX_TS_SLOPE_0_45 = -0.005

BANKROLL = 100000
KELLY_PCT = 0.2


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


def filter_to_next_50d(dates, today: datetime):
    return sorted([date for date in dates if datetime.strptime(date, "%Y-%m-%d").date() <= today.date() + timedelta(days=45) and datetime.strptime(date, "%Y-%m-%d").date() != today.date()])


def calculate_historical_volatility(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)

    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2

    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) *
              window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


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

# Extract earnings dates
upcoming_earnings = []
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
    # TODO: can the exp_dates be stored as datetime or date objects instead of strings?
    filtered_exp_dates = filter_to_next_50d(list(single_stock.options), today)
    # make sure there are at least 4 provided expiration dates for the stock in the net 50d. this is a rough proxy for "does the stock have weekly options available"
    # we do this because we want to be as close as possible to targeting a 30d spread on our calendar, and without weekly options we are unlikely to be able to do that
    # note that I also think "are weekly options available" is a rough proxy for "does this stock have alot of trading volume", which is a critera we check later anyway
    if len(filtered_exp_dates) < 4:
        print("skipped! (no weekly options)")
        continue
    # get the option chains for each of the expiration dates
    options_chains = {exp_date: single_stock.option_chain(
        exp_date) for exp_date in filtered_exp_dates}
    underlying_price = get_current_price(single_stock) if tickers_with_price_and_timestamps[symbol]["price"] is None else tickers_with_price_and_timestamps[symbol]["price"]
    if underlying_price is None:
        print("skipped! (cannot determine stock price)")
        continue
    at_the_money_impl_vols = {}
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

    if not at_the_money_impl_vols:
        #TODO: should we have a certain number of at_the_money_impl_vols instead of just checking whether it is empty?
        print("skipped! (no values for at-the-money implied volatility! (need to figure out when this occurs and how we can prevent it))")
        continue
    days_to_expiry_entries = []
    impl_vols = []
    for exp_date, impl_vol in at_the_money_impl_vols.items():
        days_to_expiry = (datetime.strptime(
            exp_date, "%Y-%m-%d").date() - today.date()).days
        days_to_expiry_entries.append(days_to_expiry)
        impl_vols.append(impl_vol)

    term_spline = build_term_structure(days_to_expiry_entries, impl_vols)

    ts_slope_0_45 = (term_spline(45) - term_spline(days_to_expiry_entries[0])) / (45-days_to_expiry_entries[0])

    price_history = single_stock.history(period='3mo')
    # TODO: make sure the name of the calcualte vol function is accurate
    iv30_rv30 = term_spline(30) / calculate_historical_volatility(price_history)

    avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

    # if all three criteria are met, then this stock can be recommended for the strategy
    if (avg_volume < MIN_30D_AVG_VOLUME):
        print("skipped! (not enough average volume)")
        continue
    if (iv30_rv30 < MIN_30D_IV_RV):
        print("skipped! (iv/rv too low)")
        continue
    if (ts_slope_0_45 > MAX_TS_SLOPE_0_45):
        print("skipped! (ts_slope_0_45 too high)")
        continue
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
    upcoming_earnings.append({"symbol": symbol, "price": round(underlying_price, 2), "earnings_call_datetime": tickers_with_price_and_timestamps[symbol][
                             "earnings_call_datetime"], "enter_position_time": tickers_with_price_and_timestamps[symbol]["enter_position_time"], "exit_position_time": tickers_with_price_and_timestamps[symbol]["exit_position_time"]})
    print("added!")

upcoming_earnings_sorted = sorted(upcoming_earnings, key=lambda event: (
    event["earnings_call_datetime"], event["enter_position_time"]))
for upcoming_earning in upcoming_earnings_sorted:
    # TODO: we could also use the options current price to tell the user how many calendars they should buy based on their bankroll amount
    print(f"{upcoming_earning["symbol"]} @ {upcoming_earning["price"]} (earnings @ {upcoming_earning["earnings_call_datetime"]}) - open @ {upcoming_earning["enter_position_time"]}, close @ {upcoming_earning["exit_position_time"]}")
