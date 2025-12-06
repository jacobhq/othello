ALTER TABLE Game
    ADD type TEXT NOT NULL CHECK (type IN ('user_user', 'user_model', 'user_anon')) default 'user_anon'
        CHECK (
            (type = 'user_user' AND player_one_id IS NOT NULL AND player_two_id IS NOT NULL)
                OR
            (type = 'user_model' AND player_one_id IS NOT NULL AND player_two_id IS NOT NULL)
                OR
            (type = 'anon' AND player_one_id IS NOT NULL AND player_two_id IS NULL)
            )