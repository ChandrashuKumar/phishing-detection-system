import {NextResponse} from "next/server";

export async function POST(req){
    const {email} = await req.json();

    if(!email){
        return NextResponse.json({error: "Missing email text"}, {status: 400});
    }

    try {
        const res = await fetch(`${process.env.BACKEND_API_URL}/api/email-sms/detect`,{
            method: "POST",
            headers:{
                "Content-Type": "application/json"
            },
            body: JSON.stringify({text:email})
        });

        const data = await res.json();

        if(!res.ok){
            return NextResponse.json({error: data.error || "Backend error"}, {status: res.status});
        }

        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({error: error.message}, {status: 500});
    }

}