-- 003: engine settings, for values the user edits at runtime.
--
-- Capital lives server-side rather than in browser storage so it survives
-- a device change and so position sizing can be computed by the engine
-- rather than duplicated in the UI.

set search_path to swing, public;

create table if not exists engine_settings (
    key         text primary key,
    value       jsonb not null,
    updated_at  timestamptz default now()
);

insert into engine_settings (key, value) values
    ('capital', '0'::jsonb),
    ('risk_pct', '2.5'::jsonb)
on conflict (key) do nothing;
