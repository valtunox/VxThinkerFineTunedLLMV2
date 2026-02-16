# RedisService Usage

## Overview
`RedisService` is an async, enterprise-ready service for advanced Redis operations in cloud environments. It supports key-value, hash, pub/sub, backup/restore, and more.

## Features
- Set, get, delete, and list keys
- Set key with expiry (TTL)
- Pattern-based key search
- Atomic increment
- Hash (dictionary) operations
- Pub/Sub (publish only)
- Backup and restore

## Example Usage

```python
from redis_service import RedisService
import asyncio

async def main():
    redis = RedisService()
    await redis.set_key('user:1', 'Alice')
    print(await redis.get_key('user:1'))  # Alice
    await redis.set_key_with_expiry('temp', 'value', ttl=60)
    await redis.set_hash('user:2', {'name': 'Bob', 'age': 30})
    print(await redis.get_hash('user:2'))  # {'name': 'Bob', 'age': 30}
    await redis.incr_key('counter')
    print(await redis.list_keys())
    await redis.publish('events', 'User signed up')
    backup = await redis.backup()
    await redis.restore(backup)

asyncio.run(main())
```

## Methods
- `set_key(key, value)`
- `get_key(key)`
- `delete_key(key)`
- `list_keys()`
- `set_key_with_expiry(key, value, ttl)`
- `find_keys(pattern)`
- `incr_key(key, amount=1)`
- `set_hash(hash_key, mapping)`
- `get_hash(hash_key)`
- `publish(channel, message)`
- `backup()`
- `restore(data)`

---
For production, connect a real Redis client and implement expiry/background tasks as needed.
