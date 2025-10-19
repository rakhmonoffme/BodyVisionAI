-- # Body Analysis Database Schema
--
-- 1. New Tables
--    - analysis_sessions: Stores analysis session data
--    - measurements: Individual body measurements
--    - body_composition: Body composition analysis results
--    - session_images: Uploaded and processed images
--
-- 2. Security
--    - Enable RLS on all tables
--    - Allow public access (no auth required for MVP)
--
-- 3. Indexes
--    - session_id for fast lookups

CREATE TABLE IF NOT EXISTS analysis_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_height numeric NOT NULL,
  user_weight numeric NOT NULL,
  user_age integer NOT NULL,
  user_gender text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  progress numeric DEFAULT 0,
  current_step text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS measurements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  measurement_type text NOT NULL,
  value numeric NOT NULL,
  reference_min numeric,
  reference_max numeric,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS body_composition (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  body_fat_percentage numeric,
  lean_mass numeric,
  fat_mass numeric,
  bmi numeric,
  body_type text,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS session_images (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id uuid NOT NULL REFERENCES analysis_sessions(id) ON DELETE CASCADE,
  image_type text NOT NULL,
  image_url text NOT NULL,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE analysis_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE body_composition ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_images ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public insert on analysis_sessions"
  ON analysis_sessions FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public read on analysis_sessions"
  ON analysis_sessions FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public update on analysis_sessions"
  ON analysis_sessions FOR UPDATE
  TO anon
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Allow public read on measurements"
  ON measurements FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert on measurements"
  ON measurements FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public read on body_composition"
  ON body_composition FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert on body_composition"
  ON body_composition FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE POLICY "Allow public read on session_images"
  ON session_images FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "Allow public insert on session_images"
  ON session_images FOR INSERT
  TO anon
  WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_measurements_session_id ON measurements(session_id);
CREATE INDEX IF NOT EXISTS idx_body_composition_session_id ON body_composition(session_id);
CREATE INDEX IF NOT EXISTS idx_session_images_session_id ON session_images(session_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON analysis_sessions(status);