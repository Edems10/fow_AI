from utils import read_matches_from_batches


class TestUtils:
    data_folder: str = 'D:/fantasyai/data'

    def test_read_matches_from_batches(self):
        version = '11.15'
        # Test gameid unity in retrieved files and uniqueness of gameids
        ids = set()
        for events in read_matches_from_batches(self.data_folder, version):
            game_ids = set(event['gameID'] for event in events)
            assert len(game_ids) == 1
            gameid = events[0]['gameID']
            if gameid in ids:
                print(F"ID found multiple times! {gameid}")
            ids.add(gameid)
            for i in range(1, len(events)):
                if events[i-1]['gameTime'] > events[i]['gameTime']:
                    print(events[i-1], events[i])
                assert events[i-1]['gameTime'] <= events[i]['gameTime']
            # Validate assumptions about redundancy
            for event in events:
                if event['rfc461Schema'] == 'item_undo':
                    assert 'itemAfterUndo' not in event or 'itemBeforeUndo' not in event
                    if 'itemAfterUndo' in event:
                        assert event['itemAfterUndo'] == event['itemID']
                    elif 'itemBeforeUndo' in event:
                        assert event['itemBeforeUndo'] == event['itemID']

