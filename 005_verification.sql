-- 005: signal verification (taken / skipped + reason).
--
-- Mirror of verification.DDL. The service applies exactly this at startup (idempotent), so nobody needs to run it by
-- hand; the file exists so the schema is reproducible from the repo.

set search_path to swing, public;

alter table signals add column if not exists verification_outcome text check (verification_outcome in ('taken','skipped'));
alter table signals add column if not exists verification_reason  text;   -- skipped only: bad_geometry | pattern_absent | bad_rr | stale_data | regime_mismatch | other
alter table signals add column if not exists verification_note    text;   -- optional free text
alter table signals add column if not exists verification_at      timestamptz;

create index if not exists signals_unreviewed_idx on signals (as_of_date) where verification_outcome is null;

update signals s
   set verification_outcome = 'taken', verification_at = coalesce(o.entry_date::timestamptz, now())
  from signal_outcomes o
 where o.signal_id = s.id and o.entry_price is not null and s.verification_outcome is null;
