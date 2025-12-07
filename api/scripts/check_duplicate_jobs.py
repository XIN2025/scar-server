import asyncio
from app.services.backend_services.db import get_db
from collections import Counter

async def check_duplicates():
    db = get_db()
    jobs = await db['nudge_scheduled_jobs'].find({}).to_list(length=None)
    
    # Get all job IDs
    ids = [j.get('id') or str(j.get('_id', '')) for j in jobs]
    
    # Count duplicates
    counter = Counter(ids)
    dupes = [job_id for job_id, count in counter.items() if count > 1]
    
    print(f'Total jobs in MongoDB: {len(jobs)}')
    print(f'Unique job IDs: {len(set(ids))}')
    print(f'Duplicate IDs found: {len(dupes)}')
    
    if dupes:
        print(f'\n⚠️  Found {len(dupes)} duplicate job IDs:')
        for dup_id in dupes[:20]:  # Show first 20
            dup_jobs = [j for j in jobs if (j.get('id') or str(j.get('_id', ''))) == dup_id]
            print(f'  - {dup_id}: {len(dup_jobs)} instances')
            # Show email if available
            for job in dup_jobs[:3]:
                if 'args' in job and job['args']:
                    print(f'    Email: {job["args"][0] if job["args"] else "N/A"}')
    else:
        print('\n✅ No duplicate job IDs found in MongoDB')
    
    # Also check by email and notification type
    print('\n--- Checking by email and notification type ---')
    email_jobs = {}
    for job in jobs:
        job_id = job.get('id') or str(job.get('_id', ''))
        if 'args' in job and job['args'] and len(job['args']) > 0:
            email = job['args'][0]
            # Extract notification type from job_id
            if job_id.startswith('morning_'):
                key = f'{email}_morning'
            elif job_id.startswith('evening_'):
                key = f'{email}_evening'
            elif job_id.startswith('night_'):
                key = f'{email}_night'
            elif job_id.startswith('checkin_'):
                key = f'{email}_checkin'
            else:
                key = f'{email}_unknown'
            
            if key not in email_jobs:
                email_jobs[key] = []
            email_jobs[key].append(job_id)
    
    dupes_by_email = {k: v for k, v in email_jobs.items() if len(v) > 1}
    if dupes_by_email:
        print(f'\n⚠️  Found {len(dupes_by_email)} email/type combinations with multiple jobs:')
        for key, job_ids in list(dupes_by_email.items())[:10]:
            print(f'  - {key}: {len(job_ids)} jobs')
            print(f'    Job IDs: {job_ids}')
    else:
        print('\n✅ No duplicate jobs by email/type combination')

if __name__ == '__main__':
    asyncio.run(check_duplicates())

