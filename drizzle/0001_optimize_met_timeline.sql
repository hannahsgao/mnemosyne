CREATE INDEX IF NOT EXISTS idx_bins_end_start
ON bins(bin_end, bin_start);
--> statement-breakpoint
PRAGMA optimize;
