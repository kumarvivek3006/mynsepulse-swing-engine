-- 004: scale-out and MA exit settings.
--
-- Scaling a portion out at the first objective and trailing the remainder
-- is the common skeleton across discretionary swing practitioners. The
-- fraction is a risk preference, not an engine constant, so it lives here.

set search_path to swing, public;

insert into engine_settings (key, value) values
    ('scale_out_pct', '50'::jsonb),      -- % of position sold at T1
    ('ma_exit_period', '20'::jsonb)      -- close below this EMA = exit signal
on conflict (key) do nothing;

-- Partial exits: a trade can now be reduced without being closed.
alter table signal_outcomes
    add column if not exists scaled_qty int,
    add column if not exists scaled_price numeric,
    add column if not exists scaled_date date;
