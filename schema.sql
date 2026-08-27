-- mynsepulse — core schema v0.2
-- Postgres / Supabase
--
-- All objects live in the `swing` schema. mynsepulse hosts multiple products
-- (intraday options, swing, bitcoin options) against one database, and table
-- names like `signals` and `symbols` would otherwise collide. Every product
-- gets its own schema; `public` is reserved for genuinely shared entities.

create extension if not exists "uuid-ossp";

create schema if not exists swing;
set search_path to swing, public;

-- Grant the Railway scanner (service_role) full access; the Lovable client
-- reaches this schema only through explicit views, never directly.
grant usage on schema swing to service_role;

-- ---------------------------------------------------------------
-- Symbol master
-- ---------------------------------------------------------------
create table symbols (
    symbol            text primary key,          -- 'RELIANCE'
    fyers_symbol      text not null unique,       -- 'NSE:RELIANCE-EQ'
    isin              text,
    company_name      text,
    series            text,                       -- EQ / BE / T2T
    sector            text,
    industry          text,
    listing_date      date,
    market_cap_cr     numeric,
    in_nifty500       boolean default false,
    in_fno            boolean default false,
    is_active         boolean default true,
    updated_at        timestamptz default now()
);

-- ---------------------------------------------------------------
-- Price data. Unadjusted is stored as received; adjusted is derived
-- so an adjustment bug is always reversible.
-- ---------------------------------------------------------------
create table ohlcv_daily (
    symbol        text not null references symbols(symbol) on delete cascade,
    trade_date    date not null,
    open          numeric not null,
    high          numeric not null,
    low           numeric not null,
    close         numeric not null,
    volume        bigint  not null,
    adj_factor    numeric not null default 1.0,
    adj_open      numeric generated always as (open  * adj_factor) stored,
    adj_high      numeric generated always as (high  * adj_factor) stored,
    adj_low       numeric generated always as (low   * adj_factor) stored,
    adj_close     numeric generated always as (close * adj_factor) stored,
    primary key (symbol, trade_date)
);
create index on ohlcv_daily (trade_date desc);

create table ohlcv_weekly (
    symbol        text not null references symbols(symbol) on delete cascade,
    week_start    date not null,
    open numeric, high numeric, low numeric, close numeric, volume bigint,
    primary key (symbol, week_start)
);

-- ---------------------------------------------------------------
-- Corporate actions — drives adj_factor
-- ---------------------------------------------------------------
create table corporate_actions (
    id            uuid primary key default uuid_generate_v4(),
    symbol        text not null references symbols(symbol) on delete cascade,
    ex_date       date not null,
    action_type   text not null,     -- split | bonus | dividend | demerger | rights
    ratio_from    numeric,
    ratio_to      numeric,
    raw_purpose   text,
    ingested_at   timestamptz default now(),
    unique (symbol, ex_date, action_type, raw_purpose)
);

-- ---------------------------------------------------------------
-- Fundamentals — Gate 2 veto inputs
-- ---------------------------------------------------------------
create table fundamentals_quarterly (
    symbol             text not null references symbols(symbol) on delete cascade,
    period_end         date not null,
    revenue            numeric,
    ebitda             numeric,
    opm_pct            numeric,
    pat                numeric,
    eps                numeric,
    debt_equity        numeric,
    roce_pct           numeric,
    operating_cashflow numeric,
    receivable_days    numeric,
    auditor_flag       text,
    source             text default 'nse_xbrl',
    primary key (symbol, period_end)
);

create table shareholding (
    symbol             text not null references symbols(symbol) on delete cascade,
    period_end         date not null,
    promoter_pct       numeric,
    promoter_pledge_pct numeric,
    fii_pct            numeric,
    dii_pct            numeric,
    primary key (symbol, period_end)
);

-- ---------------------------------------------------------------
-- Event + surveillance context
-- ---------------------------------------------------------------
create table surveillance (
    symbol        text not null references symbols(symbol) on delete cascade,
    as_of         date not null,
    list_type     text not null,     -- ASM | GSM | T2T | BE
    stage         text,
    band_pct      numeric,
    primary key (symbol, as_of, list_type)
);

create table events_calendar (
    symbol        text not null references symbols(symbol) on delete cascade,
    event_date    date not null,
    event_type    text not null,     -- results | agm | ex_date
    description   text,
    is_confirmed  boolean default true,
    primary key (symbol, event_date, event_type)
);

create table news (
    id            uuid primary key default uuid_generate_v4(),
    symbol        text references symbols(symbol) on delete cascade,
    published_at  timestamptz not null,
    source        text,
    headline      text,
    url           text,
    category      text,              -- announcement | media
    sentiment     numeric,
    unique (symbol, published_at, headline)
);

-- ---------------------------------------------------------------
-- Market regime — Gate 1, one row per session
-- ---------------------------------------------------------------
create table market_regime (
    as_of              date primary key,
    state              text not null,  -- risk_on | neutral | risk_off
    nifty_close        numeric,
    nifty_vs_20dma     numeric,
    nifty_vs_50dma     numeric,
    breadth_above_50dma numeric,
    vix                numeric,
    vix_10d_change     numeric,
    distribution_days  int,
    notes              jsonb
);

-- ---------------------------------------------------------------
-- Engine output
-- ---------------------------------------------------------------
create table signals (
    id                uuid primary key default uuid_generate_v4(),
    symbol            text not null references symbols(symbol),
    generated_at      timestamptz not null default now(),
    as_of_date        date not null,

    setup_type        text not null,      -- breakout | pullback
    pattern           text not null,      -- vcp | flat_base | cup_handle | asc_triangle

    entry_trigger     numeric not null,
    stop_loss         numeric not null,
    t1                numeric not null,
    t2                numeric,
    r_multiple_t1     numeric not null,

    qty_suggested     int,
    risk_amount       numeric,

    score_total       numeric not null,
    score_breakdown   jsonb not null,

    pivot_bar_date    date,
    base_start_date   date,
    base_low          numeric,

    regime_state      text,
    notes             jsonb,
    news_flags        jsonb,

    status            text default 'pending',  -- pending|triggered|expired|stopped|target
    expires_on        date,
    constraint rr_floor check (r_multiple_t1 >= 1.0)
);
create index on signals (as_of_date desc, score_total desc);

-- Every rejection is logged. Tuning depends on seeing what we discarded.
create table gate_log (
    id            bigserial primary key,
    as_of_date    date not null,
    symbol        text not null,
    failed_gate   text,               -- null = passed all gates
    reason_code   text,
    detail        jsonb
);
create index on gate_log (as_of_date, failed_gate);

-- Outcome tracking for weight fitting
create table signal_outcomes (
    signal_id     uuid primary key references signals(id) on delete cascade,
    entry_date    date,
    entry_price   numeric,
    exit_date     date,
    exit_price    numeric,
    exit_reason   text,               -- stop | t1 | trail | time | manual
    r_realised    numeric,
    max_favourable_r numeric,
    max_adverse_r    numeric
);

-- ---------------------------------------------------------------
-- Broker session. Exactly one row. Written ONLY by the Railway
-- scanner; every other consumer reads. Enforcing a single writer is
-- what prevents two services from invalidating each other's token.
-- ---------------------------------------------------------------
create table broker_session (
    id                int primary key default 1,
    broker            text not null default 'fyers',
    access_token      text,
    refresh_token     text,
    access_expires_at timestamptz,
    refresh_expires_at timestamptz,
    last_refreshed_at timestamptz default now(),
    refreshed_by      text,              -- hostname of the writing instance
    login_method      text,              -- refresh | totp_full
    constraint single_row check (id = 1)
);

alter table broker_session enable row level security;
-- No policies created: service_role bypasses RLS, anon/authenticated get
-- nothing. The token is never reachable from the Lovable client.

create table ingestion_runs (
    id            bigserial primary key,
    job           text not null,
    started_at    timestamptz default now(),
    finished_at   timestamptz,
    status        text,
    rows_written  int,
    error         text
);

-- ---------------------------------------------------------------
-- UI read surface.
--
-- The Lovable client queries ONLY this view. It never touches
-- ohlcv_daily: a chart pulling 3 years of candles per symbol is the
-- single most likely cause of the UI feeling slow once the intraday
-- and bitcoin products are competing for the same database.
-- Charts should request a bounded window via a dedicated RPC instead.
-- ---------------------------------------------------------------
create view swing_signals_current as
select
    s.id, s.symbol, y.company_name, y.sector,
    s.as_of_date, s.setup_type, s.pattern,
    s.entry_trigger, s.stop_loss, s.t1, s.t2, s.r_multiple_t1,
    s.qty_suggested, s.risk_amount,
    s.score_total, s.score_breakdown,
    s.regime_state, s.status, s.expires_on, s.news_flags
from signals s
join symbols y on y.symbol = s.symbol
where s.status in ('pending', 'triggered')
  and s.as_of_date >= current_date - interval '10 days';

grant select on swing_signals_current to authenticated;

-- Bounded candle fetch for charting. Default window keeps the payload
-- small enough that the base and pivot are visible without shipping
-- three years of rows to a phone.
create or replace function swing_candles(p_symbol text, p_bars int default 250)
returns table (trade_date date, o numeric, h numeric, l numeric, c numeric, v bigint)
language sql stable as $$
    select trade_date, adj_open, adj_high, adj_low, adj_close, volume
    from swing.ohlcv_daily
    where symbol = p_symbol
    order by trade_date desc
    limit least(p_bars, 750);
$$;

grant execute on function swing_candles(text, int) to authenticated;
