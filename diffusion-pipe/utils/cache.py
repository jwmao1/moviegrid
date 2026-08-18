import sqlite3
import sys
from pathlib import Path
import os
import io
import time
from collections import defaultdict

import torch


class Cache:
    def __init__(self, path: str, fingerprint: str, shard_size_gb=1):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.metadata_db = self.path / 'metadata.db'
        self.shard_size_gb = shard_size_gb
        os.makedirs(self.path, exist_ok=True)

        self.init()


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        assert isinstance(idx, int)
        shard_id, shard_index = self.items[idx]
        offset, size = self.shard_metadata[shard_id][shard_index]
        shard_path = self.path / f'shard_{shard_id}.bin'

        byte_string = None
        for attempt in range(5):
            try:
                if shard_id not in self.open_files:
                    self.open_files[shard_id] = open(shard_path, 'rb')
                f = self.open_files[shard_id]
                f.seek(offset)
                byte_string = f.read(size)
                if len(byte_string) != size:
                    raise OSError(f'cache read returned {len(byte_string)} bytes, expected {size}')
                break
            except OSError as e:
                open_file = self.open_files.pop(shard_id, None)
                if open_file is not None:
                    try:
                        open_file.close()
                    except OSError:
                        pass
                if attempt == 4:
                    raise
                sleep_seconds = 2 * (attempt + 1)
                print(
                    f'[CACHE] Read failed, retrying in {sleep_seconds}s '
                    f'(attempt {attempt + 1}/5): path={shard_path} idx={idx} '
                    f'shard={shard_id} offset={offset} size={size}: {e}',
                    flush=True,
                )
                time.sleep(sleep_seconds)

        buffer = io.BytesIO(byte_string)
        item = torch.load(buffer, map_location='cpu')
        return item


    def init(self):
        print('[CACHE] Initializing')
        # create database
        connect_kwargs = {}
        if sys.version_info >= (3, 12):
            connect_kwargs['autocommit'] = False
        self.con = sqlite3.connect(self.metadata_db, **connect_kwargs)

        # check fingerprint, clear cache if different
        self.con.execute('CREATE TABLE IF NOT EXISTS fingerprint(value)')
        existing_fingerprint = self.con.execute('SELECT value FROM fingerprint').fetchone()
        if existing_fingerprint is not None:
            existing_fingerprint = existing_fingerprint[0]
            print(f'[CACHE] Existing cache has fingerprint {existing_fingerprint}')
            if self.fingerprint != existing_fingerprint:
                print('[CACHE] Fingerprint changed, deleting existing cache files')
                self.clear()
                return
        else:
            print(f'[CACHE] Storing new fingerprint: {self.fingerprint}')
            self.con.execute('INSERT INTO fingerprint VALUES(?)', (self.fingerprint,))

        # items table, current length, next shard index
        self.con.execute('CREATE TABLE IF NOT EXISTS items(shard, shard_index)')
        self.items = self.con.execute('SELECT shard, shard_index FROM items').fetchall() or []
        max_existing_shard = -1
        for shard, _ in self.items:
            max_existing_shard = max(max_existing_shard, shard)

        self.shard_metadata = defaultdict(list)
        for table_name, in self.con.execute('SELECT name FROM sqlite_master').fetchall():
            if table_name.startswith('shard_'):
                shard_id = int(table_name.split('_')[-1])
                max_existing_shard = max(max_existing_shard, shard_id)
                for entry in self.con.execute(f'SELECT offset, size FROM {table_name}').fetchall():
                    self.shard_metadata[shard_id].append(entry)

        self.shard = max_existing_shard + 1  # current shard to write to
        self.shard_file = None
        print(f'[CACHE] Existing cache length: {len(self)}')
        self.open_files = {}

        # commit
        self.con.commit()


    def clear(self):
        '''Deletes all cache files from disk. Calls init() again.'''
        self.con.close()
        os.remove(self.metadata_db)
        for bin_path in self.path.glob('*.bin'):
            os.remove(bin_path)
        self.init()


    def create_new_shard(self):
        self.shard_file = open(self.path / f'shard_{self.shard}.bin', 'wb')
        self.shard_table = f'shard_{self.shard}'
        print(f'[CACHE] Creating new shard: {self.shard_table}')
        self.con.execute(f'CREATE TABLE {self.shard_table}(offset, size)')
        self.shard_index = 0
        self.offset = 0


    def finalize_current_shard(self):
        if self.shard_file is None:
            # no-op if already finalized
            return
        self.shard_file.close()
        self.shard_file = None
        self.shard += 1
        self.con.commit()


    def add(self, item):
        if self.shard_file is None:
            self.create_new_shard()
        buffer = io.BytesIO()
        torch.save(item, buffer)
        bytes_view = buffer.getbuffer()
        self.shard_file.write(bytes_view)

        # update items metadata
        item = (self.shard, self.shard_index)
        self.items.append(item)
        self.con.execute('INSERT INTO items VALUES(?, ?)', item)
        self.shard_index += 1

        # update shard metadata
        size = len(bytes_view)
        entry = (self.offset, size)
        self.shard_metadata[self.shard].append(entry)
        self.con.execute(f'INSERT INTO {self.shard_table} VALUES (?, ?)', entry)
        self.offset += size

        # create new shard when existing one is large enough
        current_size_gb = self.shard_file.tell() / 1_000_000_000
        if current_size_gb >= self.shard_size_gb:
            self.finalize_current_shard()
