-- 001: swap the leftover Fyers column for the Upstox instrument key.
--
-- Run this only if you already executed schema.sql v0.2, which still
-- carried `fyers_symbol` from before the broker switch. Safe to run twice.

set search_path to swing, public;

alter table symbols
    add column if not exists upstox_instrument_key text;

-- The old column was NOT NULL UNIQUE; drop it outright rather than
-- leaving a required field nothing will ever populate.
alter table symbols
    drop column if exists fyers_symbol;

-- Unique but nullable: a symbol can exist in the universe before it has
-- been resolved to an instrument key, and we want to see that gap rather
-- than have the insert fail.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'symbols_upstox_instrument_key_key'
    ) then
        alter table symbols
            add constraint symbols_upstox_instrument_key_key
            unique (upstox_instrument_key);
    end if;
end $$;

create index if not exists symbols_instrument_key_idx
    on symbols (upstox_instrument_key)
    where upstox_instrument_key is not null;
