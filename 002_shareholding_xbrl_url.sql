-- 002: store the SHP filing URL alongside each shareholding row.
--
-- NSE's shareholding summary API carries promoter/public/employee-trust
-- percentages but no pledge figure. Pledge is Column XIV of the detailed
-- SHP filing, and each summary row links to that filing's XBRL. Keeping
-- the URL means the pledge parser never has to re-crawl the index.

set search_path to swing, public;

alter table shareholding
    add column if not exists xbrl_url text,
    add column if not exists pledge_parsed_at timestamptz;
